import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# Load and prepare data
mat = scipy.io.loadmat('/home/cyl/Datasets/DataMiningCourse/RD/reaction_diffusion_big.mat')
u = mat['u']
v = mat['v']
assert u.shape == v.shape
print("Data shape:", u.shape)  # 1024x251

# Reshape data to 32x32 grid (assuming 1024 = 32x32)
grid_size = 32
u_reshaped = np.zeros((grid_size, grid_size, u.shape[1]))
v_reshaped = np.zeros((grid_size, grid_size, v.shape[1]))

# Manual reshaping without using reshape function
for t in range(u.shape[1]):  # For each time step
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            u_reshaped[i, j, t] = u[idx, t]
            v_reshaped[i, j, t] = v[idx, t]

print(f"Reshaped data: u shape = {u_reshaped.shape}, v shape = {v_reshaped.shape}")

# Apply SVD for dimensionality reduction
def apply_svd(data, n_components=50):
    """
    Apply SVD to reduce dimensionality of spatial data

    Args:
        data: Input data of shape (height, width, time)
        n_components: Number of SVD components to keep

    Returns:
        reduced_data: Data with reduced spatial dimensions
        svd_model: Fitted SVD model for later reconstruction
    """
    # Reshape to (time, height*width) without using reshape directly
    original_shape = data.shape
    height, width, time = original_shape
    data_reshaped = np.zeros((time, height * width))

    # Manual reshaping
    for t in range(time):
        for h in range(height):
            for w in range(width):
                data_reshaped[t, h * width + w] = data[h, w, t]

    # Apply SVD
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(data_reshaped)

    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

    return reduced_data, svd

# Choose number of SVD components
n_svd_components = 100
print(f"Applying SVD with {n_svd_components} components...")

# Apply SVD to u and v separately
u_reduced, u_svd = apply_svd(u_reshaped, n_components=n_svd_components)
v_reduced, v_svd = apply_svd(v_reshaped, n_components=n_svd_components)

print(f"Reduced data shapes: u = {u_reduced.shape}, v = {v_reduced.shape}")

def load_rd_data_svd(u_reduced, v_reduced):
    X, Y = [], []
    for i in range(u_reduced.shape[0] - 1):
        # Stack u and v as features
        x_combined = np.concatenate([u_reduced[i], v_reduced[i]])  # Shape: (2*n_components)
        y_combined = np.concatenate([u_reduced[i+1], v_reduced[i+1]])  # Shape: (2*n_components)

        X.append(x_combined)
        Y.append(y_combined)

    return np.array(X), np.array(Y)

X, Y = load_rd_data_svd(u_reduced, v_reduced)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")  # Should be (250, 2*n_components)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training set: X shape = {X_train.shape}, Y shape = {Y_train.shape}")
print(f"Testing set: X shape = {X_test.shape}, Y shape = {Y_test.shape}")

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

# Define the neural network models
class SVDNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=None):
        super(SVDNet, self).__init__()
        if output_size is None:
            output_size = input_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

class DeepSVDNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 512, 256], output_size=None):
        super(DeepSVDNet, self).__init__()
        if output_size is None:
            output_size = input_size

        layers = []
        prev_size = input_size

        # Create hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResidualSVDNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_blocks=4, output_size=None):
        super(ResidualSVDNet, self).__init__()
        if output_size is None:
            output_size = input_size

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([self._make_res_block(hidden_size) for _ in range(num_blocks)])

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def _make_res_block(self, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )

    def forward(self, x):
        # Input layer
        out = self.input_layer(x)

        # Residual blocks
        for res_block in self.res_blocks:
            residual = out
            out = res_block(out)
            out += residual
            out = nn.functional.relu(out)

        # Output layer
        out = self.output_layer(out)

        return out

class UNetSVDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetSVDBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class UNetSVD(nn.Module):
    def __init__(self, input_size, output_size=None):
        super(UNetSVD, self).__init__()
        if output_size is None:
            output_size = input_size

        # Encoder path
        self.enc1 = UNetSVDBlock(input_size, 256)
        self.enc2 = UNetSVDBlock(256, 512)
        self.enc3 = UNetSVDBlock(512, 1024)

        # Bottleneck
        self.bottleneck = UNetSVDBlock(1024, 2048)

        # Decoder path
        self.dec3 = UNetSVDBlock(2048 + 1024, 1024)  # +1024 for skip connection
        self.dec2 = UNetSVDBlock(1024 + 512, 512)    # +512 for skip connection
        self.dec1 = UNetSVDBlock(512 + 256, 256)     # +256 for skip connection

        # Final output layer
        self.final = nn.Linear(256, output_size)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Decoder with skip connections
        dec3 = self.dec3(torch.cat([bottleneck, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))

        # Final output
        output = self.final(dec1)

        return output

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Testing
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

        epoch_test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}")

    return model, train_losses, test_losses

# Visualization function for SVD-based model
def visualize_svd_prediction(model, X_test_tensor, Y_test_tensor, u_svd, v_svd, idx=0, save_dir='p2'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        input_sample = X_test_tensor[idx:idx+1].to(device)
        true_output = Y_test_tensor[idx].cpu().numpy()
        predicted_output = model(input_sample).cpu().numpy()[0]

    # Split the output into u and v components
    n_components = true_output.shape[0] // 2
    true_u_reduced = true_output[:n_components]
    true_v_reduced = true_output[n_components:]
    pred_u_reduced = predicted_output[:n_components]
    pred_v_reduced = predicted_output[n_components:]

    # Reconstruct the original spatial dimensions (32x32)
    # Using inverse_transform without explicit reshaping
    true_u_flat = u_svd.inverse_transform(true_u_reduced.reshape(1, -1))
    true_v_flat = v_svd.inverse_transform(true_v_reduced.reshape(1, -1))
    pred_u_flat = u_svd.inverse_transform(pred_u_reduced.reshape(1, -1))
    pred_v_flat = v_svd.inverse_transform(pred_v_reduced.reshape(1, -1))

    # Convert to 32x32 grid
    true_u_reconstructed = true_u_flat.reshape(32, 32)
    true_v_reconstructed = true_v_flat.reshape(32, 32)
    pred_u_reconstructed = pred_u_flat.reshape(32, 32)
    pred_v_reconstructed = pred_v_flat.reshape(32, 32)

    # Upscale to 512x512 using bicubic interpolation
    # Create PIL images for upscaling
    true_u_img = Image.fromarray(true_u_reconstructed)
    true_v_img = Image.fromarray(true_v_reconstructed)
    pred_u_img = Image.fromarray(pred_u_reconstructed)
    pred_v_img = Image.fromarray(pred_v_reconstructed)

    # Resize to 512x512
    true_u_upscaled = np.array(true_u_img.resize((512, 512), Image.BICUBIC))
    true_v_upscaled = np.array(true_v_img.resize((512, 512), Image.BICUBIC))
    pred_u_upscaled = np.array(pred_u_img.resize((512, 512), Image.BICUBIC))
    pred_v_upscaled = np.array(pred_v_img.resize((512, 512), Image.BICUBIC))

    # Calculate errors for the upscaled images
    u_error_upscaled = np.abs(true_u_upscaled - pred_u_upscaled)
    v_error_upscaled = np.abs(true_v_upscaled - pred_v_upscaled)

    # Create a figure with subplots for u and v components
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot u component (512x512)
    im0 = axes[0, 0].imshow(true_u_upscaled, cmap='viridis')
    axes[0, 0].set_title('True u (512x512)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(pred_u_upscaled, cmap='viridis')
    axes[0, 1].set_title('Predicted u (512x512)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(u_error_upscaled, cmap='hot')
    axes[0, 2].set_title('Absolute Error u (512x512)')
    plt.colorbar(im2, ax=axes[0, 2])

    # Plot v component (512x512)
    im3 = axes[1, 0].imshow(true_v_upscaled, cmap='viridis')
    axes[1, 0].set_title('True v (512x512)')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(pred_v_upscaled, cmap='viridis')
    axes[1, 1].set_title('Predicted v (512x512)')
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(v_error_upscaled, cmap='hot')
    axes[1, 2].set_title('Absolute Error v (512x512)')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'prediction_visualization_512x512.png')
    plt.savefig(save_path, dpi=300)
    plt.show()

    # Also save the original 32x32 visualization for comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot u component (32x32)
    im0 = axes[0, 0].imshow(true_u_reconstructed, cmap='viridis')
    axes[0, 0].set_title('True u (32x32)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(pred_u_reconstructed, cmap='viridis')
    axes[0, 1].set_title('Predicted u (32x32)')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(np.abs(true_u_reconstructed - pred_u_reconstructed), cmap='hot')
    axes[0, 2].set_title('Absolute Error u (32x32)')
    plt.colorbar(im2, ax=axes[0, 2])

    # Plot v component (32x32)
    im3 = axes[1, 0].imshow(true_v_reconstructed, cmap='viridis')
    axes[1, 0].set_title('True v (32x32)')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(pred_v_reconstructed, cmap='viridis')
    axes[1, 1].set_title('Predicted v (32x32)')
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(np.abs(true_v_reconstructed - pred_v_reconstructed), cmap='hot')
    axes[1, 2].set_title('Absolute Error v (32x32)')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'prediction_visualization_32x32.png')
    plt.savefig(save_path)
    plt.show()

    # Save the prediction data (both 32x32 and 512x512)
    # 32x32 data - flatten only at the end for saving
    u_prediction_data_32x32 = {
        'True_u': true_u_reconstructed.flatten(),
        'Predicted_u': pred_u_reconstructed.flatten(),
        'Absolute_Error_u': np.abs(true_u_reconstructed - pred_u_reconstructed).flatten()
    }
    v_prediction_data_32x32 = {
        'True_v': true_v_reconstructed.flatten(),
        'Predicted_v': pred_v_reconstructed.flatten(),
        'Absolute_Error_v': np.abs(true_v_reconstructed - pred_v_reconstructed).flatten()
    }

    # 512x512 data (sampled to reduce file size)
    # Use advanced indexing instead of reshape+sample
    indices = np.random.choice(512*512, size=10000, replace=False)
    u_prediction_data_512x512 = {
        'True_u': true_u_upscaled.flat[indices],
        'Predicted_u': pred_u_upscaled.flat[indices],
        'Absolute_Error_u': u_error_upscaled.flat[indices]
    }
    v_prediction_data_512x512 = {
        'True_v': true_v_upscaled.flat[indices],
        'Predicted_v': pred_v_upscaled.flat[indices],
        'Absolute_Error_v': v_error_upscaled.flat[indices]
    }

    # Save to CSV
    pd.DataFrame(u_prediction_data_32x32).to_csv(os.path.join(save_dir, 'u_prediction_data_32x32.csv'), index=False)
    pd.DataFrame(v_prediction_data_32x32).to_csv(os.path.join(save_dir, 'v_prediction_data_32x32.csv'), index=False)
    pd.DataFrame(u_prediction_data_512x512).to_csv(os.path.join(save_dir, 'u_prediction_data_512x512.csv'), index=False)
    pd.DataFrame(v_prediction_data_512x512).to_csv(os.path.join(save_dir, 'v_prediction_data_512x512.csv'), index=False)

    # Calculate and save error metrics
    error_metrics = {
        'u_32x32_mae': np.mean(np.abs(true_u_reconstructed - pred_u_reconstructed)),
        'v_32x32_mae': np.mean(np.abs(true_v_reconstructed - pred_v_reconstructed)),
        'u_32x32_mse': np.mean((true_u_reconstructed - pred_u_reconstructed)**2),
        'v_32x32_mse': np.mean((true_v_reconstructed - pred_v_reconstructed)**2),
        'u_512x512_mae': np.mean(u_error_upscaled),
        'v_512x512_mae': np.mean(v_error_upscaled),
        'u_512x512_mse': np.mean(u_error_upscaled**2),
        'v_512x512_mse': np.mean(v_error_upscaled**2)
    }

    with open(os.path.join(save_dir, 'error_metrics.txt'), 'w') as f:
        for metric, value in error_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")

    return error_metrics

# Main execution
if __name__ == "__main__":
    # Create the p2 directory if it doesn't exist
    save_dir = 'p2_svd'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Get the input size (2 * n_svd_components)
    input_size = X_train_tensor.shape[1]

    # Choose one of the models
    # model = SVDNet(input_size=input_size)
    # model = DeepSVDNet(input_size=input_size)
    # model = ResidualSVDNet(input_size=input_size)
    model = UNetSVD(input_size=input_size)
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
    error_metrics = visualize_svd_prediction(trained_model, X_test_tensor, Y_test_tensor, u_svd, v_svd, save_dir=save_dir)

    # Save the model
    model_path = os.path.join(save_dir, f'{model_name}_prediction_model.pth')
    torch.save(trained_model.state_dict(), model_path)

    # Save model architecture information
    with open(os.path.join(save_dir, 'model_info.txt'), 'w') as f:
        f.write(f"Model Architecture: {model_name}\n")
        f.write(f"SVD Components: {n_svd_components}\n")
        f.write(f"Input Size: {input_size} (2 * {n_svd_components})\n")
        f.write(f"Output Size: {input_size} (2 * {n_svd_components})\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Testing Loss: {test_losses[-1]:.6f}\n")
        f.write(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"U Explained Variance: {u_svd.explained_variance_ratio_.sum():.4f}\n")
        f.write(f"V Explained Variance: {v_svd.explained_variance_ratio_.sum():.4f}\n")
        f.write("\nError Metrics:\n")
        for metric, value in error_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")

    print(f"All files saved successfully to the '{save_dir}' directory!")
    print(f"Model saved as: {model_path}")
    print(f"Loss curves saved as: {os.path.join(save_dir, 'loss_curves.png')}")
    print(f"Prediction visualizations saved as:")
    print(f"  - {os.path.join(save_dir, 'prediction_visualization_32x32.png')}")
    print(f"  - {os.path.join(save_dir, 'prediction_visualization_512x512.png')}")
    print(f"Loss data saved as: {os.path.join(save_dir, 'loss_data.csv')}")
    print(f"Prediction data saved as:")
    print(f"  - {os.path.join(save_dir, 'u_prediction_data_32x32.csv')}")
    print(f"  - {os.path.join(save_dir, 'v_prediction_data_32x32.csv')}")
    print(f"  - {os.path.join(save_dir, 'u_prediction_data_512x512.csv')}")
    print(f"  - {os.path.join(save_dir, 'v_prediction_data_512x512.csv')}")
    print(f"Error metrics saved as: {os.path.join(save_dir, 'error_metrics.txt')}")
    print(f"Model information saved as: {os.path.join(save_dir, 'model_info.txt')}")
