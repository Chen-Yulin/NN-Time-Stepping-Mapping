import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

# Load and prepare data
mat_0 = scipy.io.loadmat('data/KS/kuramoto_sivishinky_0.mat')
mat_1 = scipy.io.loadmat('data/KS/kuramoto_sivishinky_1.mat')

# Extract data from both mat files
uu_0 = mat_0['uu']  # First dataset
uu_1 = mat_1['uu']  # Second dataset

uu = np.concatenate([uu_0, uu_1], axis=1)  # Concatenate along time dimension
print("Combined data shape:", uu.shape)

def load_ks_data(uu):
    X, Y = [], []
    for i in range(uu.shape[1] - 1):
        X.append(uu[:, i])    # u(t)
        Y.append(uu[:, i + 1])  # u(t+dt)
    return np.array(X), np.array(Y)

X_0, Y_0 = load_ks_data(uu_0)
X_1, Y_1 = load_ks_data(uu_1)
X, Y = np.concatenate([X_0, X_1], axis=0), np.concatenate([Y_0, Y_1], axis=0)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train)
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.FloatTensor(Y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define neural network architectures

# 1. Simple Fully Connected Network
class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size=2048):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, input_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self, input_size):
        super(ConvNet, self).__init__()
        self.input_size = input_size

        # Reshape input to 1D spatial dimension for CNN
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape for 1D convolution [batch, 1, spatial_dim]
        x = x.view(-1, 1, self.input_size)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)

        # Reshape back to original dimensions
        return x.view(-1, self.input_size)

# 3. Residual Network
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_size, num_blocks=4):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.channels = 64

        self.conv1 = nn.Conv1d(1, self.channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.relu = nn.ReLU()

        self.res_blocks = nn.ModuleList([ResBlock(self.channels) for _ in range(num_blocks)])

        self.conv_out = nn.Conv1d(self.channels, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # Reshape for 1D convolution
        x = x.view(-1, 1, self.input_size)

        x = self.relu(self.bn1(self.conv1(x)))

        for block in self.res_blocks:
            x = block(x)

        x = self.conv_out(x)

        # Reshape back to original dimensions
        return x.view(-1, self.input_size)

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Testing
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

        epoch_test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}")

    return model, train_losses, test_losses

# Visualization function
def visualize_prediction(model, X_test_tensor, Y_test_tensor, idx=0, save_dir='p1'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        input_sample = X_test_tensor[idx:idx+1].to(device)
        true_output = Y_test_tensor[idx].cpu().numpy()
        predicted_output = model(input_sample).cpu().numpy()[0]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(true_output, label='True')
    plt.plot(predicted_output, label='Predicted')
    plt.legend()
    plt.title('True vs Predicted')

    plt.subplot(1, 2, 2)
    plt.plot(np.abs(true_output - predicted_output))
    plt.title('Absolute Error')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'prediction_visualization.png')
    plt.savefig(save_path)
    plt.show()

    # Save the prediction data
    prediction_data = {
        'True': true_output,
        'Predicted': predicted_output,
        'Absolute_Error': np.abs(true_output - predicted_output)
    }
    prediction_df = pd.DataFrame(prediction_data)
    prediction_df.to_csv(os.path.join(save_dir, 'prediction_data.csv'), index=False)

# Main execution
if __name__ == "__main__":
    # Create the p1 directory if it doesn't exist
    save_dir = 'p1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    input_size = X_train.shape[1]

    # Choose one of the models
    # model = FCNet(input_size)
    # model = ConvNet(input_size)
    model = ResNet(input_size)
    model_name = model.__class__.__name__

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model, train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=30
    )

    # Save loss values to CSV
    loss_df = pd.DataFrame({
        'Epoch': list(range(1, len(train_losses) + 1)),
        'Training_Loss': train_losses,
        'Testing_Loss': test_losses
    })
    loss_df.to_csv(os.path.join(save_dir, 'loss_data.csv'), index=False)

    # Plot training and testing loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Testing Loss - {model_name}')
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.show()

    # Visualize predictions
    visualize_prediction(trained_model, X_test_tensor, Y_test_tensor, save_dir=save_dir)

    # Save the model
    model_path = os.path.join(save_dir, f'{model_name}_prediction_model.pth')
    torch.save(trained_model.state_dict(), model_path)

    # Save model architecture information
    with open(os.path.join(save_dir, 'model_info.txt'), 'w') as f:
        f.write(f"Model Architecture: {model_name}\n")
        f.write(f"Input Size: {input_size}\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Testing Loss: {test_losses[-1]:.6f}\n")
        f.write(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print(f"All files saved successfully to the '{save_dir}' directory!")
    print(f"Model saved as: {model_path}")
    print(f"Loss curves saved as: {os.path.join(save_dir, 'loss_curves.png')}")
    print(f"Prediction visualization saved as: {os.path.join(save_dir, 'prediction_visualization.png')}")
    print(f"Loss data saved as: {os.path.join(save_dir, 'loss_data.csv')}")
    print(f"Prediction data saved as: {os.path.join(save_dir, 'prediction_data.csv')}")
    print(f"Model information saved as: {os.path.join(save_dir, 'model_info.txt')}")
