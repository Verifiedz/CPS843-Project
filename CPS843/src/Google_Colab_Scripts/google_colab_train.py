import os
import sys
import glob
import zipfile
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.cuda.empty_cache()


from google.colab import drive
drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/Colab Notebooks/data.zip'
extract_path = '/content/data'
save_path = '/content/drive/MyDrive/Colab Notebooks/AOD_NET_HYPER_TUNED.pth'

print(f"Reading zip from: {zip_path}")
print(f"Extracting to temp folder: {extract_path}")

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

if not os.path.exists(os.path.join(extract_path, 'data', 'raw', 'train')):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(" Data successfully extracted to temp folder.")
    except FileNotFoundError:
        print(" ERROR: Could not find data.zip. Please check path.")
else:
    print(" Data already extracted. Skipping unzip.")


class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)

        k = torch.sigmoid(self.conv5(cat3))

        clean_prediction = k * x - k + 1.0
        clean_prediction = F.relu(clean_prediction)
        clean_prediction = torch.min(clean_prediction, torch.tensor(1.0).to(x.device))

        return clean_prediction

class SotsDataset(Dataset):
    def __init__(self, data_path):
        self.hazy_files = sorted(glob.glob(os.path.join(data_path, 'hazy') + '/*'))
        self.clear_files = sorted(glob.glob(os.path.join(data_path, 'clear') + '/*'))

    def __getitem__(self, index):
        hazy_img = Image.open(self.hazy_files[index]).convert('RGB')
        clear_img = Image.open(self.clear_files[index]).convert('RGB')

        hazy_img = hazy_img.resize((640, 480), Image.Resampling.LANCZOS)
        clear_img = clear_img.resize((640, 480), Image.Resampling.LANCZOS)

        hazy_tensor = torch.from_numpy(np.array(hazy_img)).float().permute(2, 0, 1) / 255.0
        clear_tensor = torch.from_numpy(np.array(clear_img)).float().permute(2, 0, 1) / 255.0
        return hazy_tensor, clear_tensor

    def __len__(self):
        return len(self.hazy_files)

def get_loader(data_path, batch_size):
    return DataLoader(dataset=SotsDataset(data_path), batch_size=batch_size, shuffle=True, num_workers=2)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AODNet().to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=5e-3, eps=1e-5)

    train_path = '/content/data/data/raw/train'
    print(f"Loading Dataset from temp folder: {train_path}")

    try:
        train_loader = get_loader(train_path, batch_size=8)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Commencing KICKSTART Training (15 Epochs)...")

    for epoch in range(50):
        model.train()
        epoch_loss = 0

        for i, (hazy, clear) in enumerate(train_loader):
            hazy = hazy.to(device)
            clear = clear.to(device)

            optimizer.zero_grad()
            outputs = model(hazy)

            if epoch == 0 and i == 0:
                diff = torch.abs(outputs - hazy).mean().item()
                print(f"--- CHECK: First Batch Difference: {diff:.6f} (Should be > 0) ---")

            loss = criterion(outputs, clear)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/15 complete. Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print("------------------------------------------------")
    print(f"Training Complete. Model saved to: {save_path}")
    print("------------------------------------------------")

if __name__ == "__main__":
    train_model()