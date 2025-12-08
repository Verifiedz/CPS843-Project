import cv2
import torch
import argparse
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from src.models.aod_net import AODNet

def parse_args():
    parser = argparse.ArgumentParser(description="Image Dehazing Pipeline (DCP & AOD-Net)")
    parser.add_argument('--input', type=str, default='data/raw/test/hazy/sample.jpg', help='Path to input image')
    parser.add_argument('--method', type=str, default='aod', choices=['dcp', 'aod', 'all'], help='Dehazing method to use')
    parser.add_argument('--output', type=str, default='outputs/images/', help='Directory to save results')
    return parser.parse_args()

def check_environment():
    print(f"--- Environment Check ---")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}") 
    else:
        print("WARNING: CUDA not available. Running on CPU.")
    print("-------------------------")

def run_aod(image_path, output_dir):
   
    print(f"[AOD-Net] Starting inference on {image_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AODNet().to(device)
    weights_path = 'outputs/aod_net.pth'

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("TIP: Run 'python -m src.models.train' first to generate the model file.")
        return

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval() 
        print("[AOD-Net] Weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load weights. {e}")
        return

    try:
        img = Image.open(image_path).convert('RGB')
    except IOError:
        print(f"ERROR: Could not open image {image_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device) 

    with torch.no_grad():
        clean_tensor = model(img_tensor)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    input_filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"AOD_result_{input_filename}")
    
    from torchvision.utils import save_image
    save_image(clean_tensor, save_path)
    
    print(f"[AOD-Net] Success! Result saved to: {save_path}")


def main():
    args = parse_args()
    check_environment()

    if not os.path.exists(args.input):
        print(f"Error: Could not find image at {args.input}")
        print("Tip: Try pointing to a file in 'data/raw/test/hazy/'")
        return

    if args.method == 'aod':
        run_aod(args.input, args.output)
        
    elif args.method == 'dcp':
        print("DCP method is currently under development by Dev 1.")
        
    elif args.method == 'all':
        run_aod(args.input, args.output)
        print("Skipping DCP (Not implemented)")

    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()