import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.metrics as ssim


hazy_dir = '/content/data/data/raw/test/hazy'
clear_dir = '/content/data/data/raw/test/clear'
weights_path = '/content/drive/MyDrive/Colab Notebooks/AOD_NET_HYPER_TUNED.pth'
output_dir = '/content/actual_evaluations_triplet'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on: {device}")

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


def calculate_psnr(img1, img2):
    """Calculates PSNR between two 0-1 range numpy arrays"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_ssim_score(img1, img2):
    """Calculates SSIM between two 0-1 range numpy arrays"""
    return ssim(img1, img2, data_range=1.0, channel_axis=2)

def load_and_preprocess(path):
    """Opens image, resizes to 640x480, converts to tensor"""
    img = Image.open(path).convert('RGB')
    img = img.resize((640, 480), Image.Resampling.LANCZOS)
    return img


model = AODNet().to(device)
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f" Weights loaded from {weights_path}")
else:
    print(f" Error: Weights not found at {weights_path}")
model.eval()



hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
target_hazy_files = hazy_files[:3] 

print(f"Processing actual predictions for: {target_hazy_files}")

for i, h_filename in enumerate(target_hazy_files):
    img_id = h_filename.split('_')[0]
    possible_clear = glob.glob(os.path.join(clear_dir, f"{img_id}.*"))
    
    if not possible_clear:
        print(f"Skipping {img_id}: No matching clear image found.")
        continue
        
    c_full_path = possible_clear[0]
    h_full_path = os.path.join(hazy_dir, h_filename)
    
    hazy_pil = load_and_preprocess(h_full_path)
    clear_pil = load_and_preprocess(c_full_path)
    
    hazy_np = np.array(hazy_pil) / 255.0
    clear_np = np.array(clear_pil) / 255.0
    
    hazy_tensor = torch.from_numpy(hazy_np).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_tensor = model(hazy_tensor)
    
    prediction_np = prediction_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    baseline_psnr = calculate_psnr(hazy_np, clear_np)
    actual_psnr = calculate_psnr(prediction_np, clear_np)
    actual_ssim = calculate_ssim_score(prediction_np, clear_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(hazy_np)
    axes[0].set_title("Input (Hazy)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(10, hazy_np.shape[0] - 20, f"Baseline PSNR: {baseline_psnr:.2f} dB", 
                 color='white', fontsize=10, backgroundcolor='black')

    axes[1].imshow(prediction_np)
    axes[1].set_title("AOD-Net Prediction (Actual)", fontsize=12, fontweight='bold', color='blue')
    axes[1].axis('off')
    
    axes[1].text(10, prediction_np.shape[0] - 50, f"PSNR: {actual_psnr:.2f} dB", 
                 color='lime', fontsize=10, fontweight='bold', backgroundcolor='black')
    axes[1].text(10, prediction_np.shape[0] - 20, f"SSIM: {actual_ssim:.2f}", 
                 color='lime', fontsize=10, fontweight='bold', backgroundcolor='black')

    axes[2].imshow(clear_np)
    axes[2].set_title("Ground Truth (Clear)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    save_name = f"actual_triplet_{i}.png"
    save_full_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_full_path, dpi=300)
    plt.close()
    
    print(f" Generated Actual Result: {save_full_path} (PSNR: {actual_psnr:.2f} dB)")

print(f"\nProcessing Complete. Images saved in: {output_dir}")