import torch
import torch.nn as nn
from torchvision.utils import save_image
from src.models.aod_net import AODNet
from src.utils.dataloader import get_loader
import os
import math

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio (dB)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def evaluate():
    device = torch.device("cpu")
    print(f"Evaluation Device: {device}")

    model = AODNet().to(device)
    
    weights_path = r'C:\Users\knobu\Documents\AOD_NET_GPU.pth'

    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Error: {weights_path} not found. Train the model first!")
        return

    model.eval()
    
    print("Loading SOTS-Outdoor Test Set...")
    try:
        test_loader = get_loader('data/raw/test/', batch_size=1, shuffle=False)
    except FileNotFoundError:
        print("Error: Test data not found in 'data/raw/test/'. Check your folders.")
        return
    
    avg_psnr = 0
    count = 0
    
    if not os.path.exists('outputs/images'):
        os.makedirs('outputs/images')

    print("Starting Evaluation Loop...")
    
    with torch.no_grad():
        for i, (hazy, clear) in enumerate(test_loader):
            hazy = hazy.to(device)
            clear = clear.to(device)
            
            clean_prediction = model(hazy)
            
            psnr = calculate_psnr(clean_prediction, clear)
            avg_psnr += psnr
            count += 1
            
            if i < 3:
                comparison = torch.cat((hazy, clean_prediction, clear), dim=3)
                save_path = f"outputs/images/eval_{i}.jpg"
                save_image(comparison, save_path)
                print(f"Saved visual comparison: {save_path} (PSNR: {psnr:.2f} dB)")

    if count > 0:
        final_psnr = avg_psnr / count
        print("------------------------------------------------")
        print(f"Evaluated {count} images.")
        print(f"Final Average PSNR: {final_psnr:.2f} dB")
        print("------------------------------------------------")
    else:
        print("No images were evaluated.")

if __name__ == "__main__":
    evaluate()