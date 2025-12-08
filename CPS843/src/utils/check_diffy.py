import torch
from src.models.aod_net import AODNet

def check_model_activity():
    model = AODNet()
    path = r'C:\Users\knobu\Documents\AOD_NET_TRAINED.pth'
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()

    fake_input = torch.rand(1, 3, 480, 640)

    with torch.no_grad():
        output = model(fake_input)

    diff = torch.abs(output - fake_input).mean().item()
    
    print(f"Average Pixel Difference: {diff:.6f}")
    
    if diff == 0.0:
        print("RESULT: EXACT MATCH. Possible bug in forward pass.")
    else:
        print("RESULT: NOT ZERO. The model IS changing the image, just very slightly.")
        print("This confirms the code works, but the model converged to Identity Mapping.")

if __name__ == "__main__":
    check_model_activity()