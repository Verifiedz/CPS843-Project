import torch
import torch.optim as optim
import torch.nn as nn
from src.models.aod_net import AODNet
from src.utils.dataloader import get_loader
import os

def train_model():
    device = torch.device("cpu")
    print(f"Device: {device}")

    model = AODNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    SAFE_DIR = 'C:/Users/knobu/Documents'
    FINAL_MODEL_NAME = 'AOD_NET_TRAINED.pth'

    print("Loading Dataset from data/raw/train/...")
    try:
        train_loader = get_loader('data/raw/train/', batch_size=8)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Commencing training loop (5 Epochs)...")
    
    for epoch in range(5): 
        print(f"--- Starting Epoch {epoch+1} ---")
        for i, (hazy, clear) in enumerate(train_loader):
            hazy = hazy.to(device)
            clear = clear.to(device)
            
            optimizer.zero_grad()
            outputs = model(hazy)
            loss = criterion(outputs, clear)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")
        epoch_save_path = f'{SAFE_DIR}/aod_net_epoch_{epoch}.pth'
        torch.save(model.state_dict(), epoch_save_path)
        
    final_save_path = f'{SAFE_DIR}/{FINAL_MODEL_NAME}'
    torch.save(model.state_dict(), final_save_path)
    print("------------------------------------------------")
    print("Training Complete.")
    print(f"Model successfully saved to: {final_save_path}")
    print("------------------------------------------------")
if __name__ == "__main__":
    train_model()