""" 
Copyright (c) 2025. All rights reserved.
RotNet-FA-Image: Rotation Prediction for FA Maps
Author: Yehyun  Suh
Description: This script implements a deep learning model using PyTorch for predicting rotation angles of 3D fractional anisotropy maps.
License: This script is provided "as-is" without warranty of any kind. Use at your own risk.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
import matplotlib.pyplot as plt

class NiiDataset(Dataset):
    """
    Custom Dataset to load and preprocess NIfTI (.nii.gz) files for training.

    Args:
        data_dir (str): Directory containing .nii.gz files.

    Raises:
        ValueError: If no .nii.gz files are found in the specified directory.
    """
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.files = [
            os.path.join(root, f)
            for root, _, filenames in os.walk(data_dir)
            for f in filenames if f.endswith('.nii.gz')
        ]
        if not self.files:
            raise ValueError(f"No .nii.gz files found in {data_dir}.")

        # Generate x-axis rotation angles for each sample
        self.samples = [(file_path, np.random.uniform(0, 360)) for file_path in self.files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, angle = self.samples[idx]

        try:
            img = nib.load(file_path).get_fdata()
            img = self.preprocess_fa_map(img)
            if img is None:
                return torch.full((1, 64, 64, 64), float('nan')), torch.tensor([angle])

            img_rotated = self.rotate_image(img, angle)
            return torch.tensor(img_rotated, dtype=torch.float32).unsqueeze(0), torch.tensor([angle], dtype=torch.float32)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return torch.full((1, 64, 64, 64), float('nan')), torch.tensor([angle])

    @staticmethod
    def preprocess_fa_map(img):
        """
        Preprocess an FA map by normalizing intensities and resizing to 64x64x64.

        Args:
            img (numpy.ndarray): Input 3D image.

        Returns:
            numpy.ndarray: Preprocessed 3D image or None if invalid.
        """
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        if np.all(img == 0) or np.max(img) - np.min(img) < 1e-6:
            return None

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(64, 64, 64), mode='trilinear', align_corners=False)
        img = img.squeeze().numpy()

        p1, p99 = np.percentile(img[img > 0], [1, 99])
        img = np.clip(img, p1, p99)
        return (img - p1) / (p99 - p1)

    @staticmethod
    def rotate_image(img, angle):
        """
        Rotate a 3D image along the x-axis.

        Args:
            img (numpy.ndarray): Input 3D image.
            angle (float): Rotation angle in degrees.

        Returns:
            numpy.ndarray: Rotated 3D image.
        """
        return rotate(img, angle=angle, axes=(1, 2), reshape=False)


class RotationRegressionModel(nn.Module):
    """
    3D Convolutional Neural Network for regression of rotation angles.
    """
    def __init__(self):
        super(RotationRegressionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.fc_size = self._calculate_fc_size()

        self.fc1 = nn.Linear(self.fc_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def _calculate_fc_size(self):
        """
        Calculate the size of the flattened feature vector from convolutional layers.

        Returns:
            int: Flattened feature vector size.
        """
        dummy_input = torch.zeros(1, 1, 64, 64, 64)
        output = self.conv_layers(dummy_input)
        return output.numel()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 64, 64, 64).

        Returns:
            torch.Tensor: Predicted rotation angles of shape (batch_size, 1).
        """
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_size)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        return self.fc4(x)


def train_model(data_dir, batch_size=8, epochs=20, lr=1e-3):
    """
    Train the rotation regression model on the given dataset.

    Args:
        data_dir (str): Directory containing NIfTI files.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimizer.

    Returns:
        model (RotationRegressionModel): Trained model.
        list: Training loss for each epoch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NiiDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = RotationRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, angles in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            if torch.isnan(inputs).any():
                continue

            inputs, angles = inputs.to(device), angles.to(device)
            optimizer.zero_grad()
            predicted_angles = model(inputs)
            loss = criterion(predicted_angles, angles)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model, train_losses


if __name__ == "__main__":
    DATA_DIR = "data"
    BATCH_SIZE = 18
    EPOCHS = 100
    LR = 1e-3

    model, losses = train_model(DATA_DIR, BATCH_SIZE, EPOCHS, LR)

    # Save final model weights
    weights_fc4 = model.fc4.weight.detach().cpu().numpy().T
    np.save("vectors.npy", weights_fc4)
