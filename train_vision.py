import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from vision_encoder import VisionEncoder

# ─── Custom Loss Function ────────────────────────────────────────

def weighted_mse_loss(reconstruction, target):
    """
    Forces the network to care 10x more about the snake and food 
    than the empty background space.
    """
    # Empty space is 0.0. Food is -1.0, Head is 1.0, Body is 0.5.
    weight_mask = torch.where(target != 0.0, 10.0, 1.0)
    squared_error = (reconstruction - target) ** 2
    return torch.mean(weight_mask * squared_error)

# ─── Training Loop ───────────────────────────────────────────────

def train_autoencoder(data_path="snake_vision_data.npy", epochs=50, batch_size=64):
    print(f"Loading dataset from {data_path}...")
    dataset_np = np.load(data_path)
    
    # Convert NumPy array to PyTorch tensors
    tensor_x = torch.tensor(dataset_np, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = VisionEncoder(latent_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Starting training on {len(dataset_np)} samples...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            inputs = batch[0]
            
            # Forward pass
            optimizer.zero_grad()
            reconstructions, _ = model(inputs)
            
            # Calculate loss and optimize
            loss = weighted_mse_loss(reconstructions, inputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Weighted MSE Loss: {avg_loss:.4f}")

    # Save the encoder so your Bipartite Graph can use it later
    os.makedirs("weights", exist_ok=True)
    encoder_path = "weights/vision_encoder.pth"
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"\nTraining complete! Encoder weights saved to '{encoder_path}'.")
    
    return model

if __name__ == "__main__":
    train_autoencoder()
