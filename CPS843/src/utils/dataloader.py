import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OutdoorDataset(Dataset):
    def __init__(self, root_dir, size=256):
        """
        Args:
            root_dir (string): Path to 'data/raw/train' or 'data/raw/test'
            size (int): Image resize dimension
        """
        self.root_dir = root_dir
        self.hazy_dir = os.path.join(root_dir, 'hazy')
        self.clear_dir = os.path.join(root_dir, 'clear')   
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor() 
        ])
        
        self.hazy_images = [x for x in os.listdir(self.hazy_dir) if x.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_name = self.hazy_images[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        if "_" in hazy_name and any(char.isdigit() for char in hazy_name.split('_')[1]):
            img_id = hazy_name.split('_')[0]
            if os.path.exists(os.path.join(self.clear_dir, f"{img_id}.png")):
                clear_name = f"{img_id}.png"
            else:
                clear_name = f"{img_id}.jpg"
        elif "hazy" in hazy_name:
            prefix = hazy_name.split('_')[0] 
            
            potential_matches = [f for f in os.listdir(self.clear_dir) if f.startswith(f"{prefix}_")]
            if potential_matches:
                clear_name = potential_matches[0]
            else:
                clear_name = hazy_name
        
        else:
            clear_name = hazy_name

        clear_path = os.path.join(self.clear_dir, clear_name)

        if not os.path.exists(clear_path):
            raise FileNotFoundError(f"Unable to find corresponding clear image {hazy_name}. Checked: {clear_path}")

        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        return self.transform(hazy_img), self.transform(clear_img)

def get_loader(root_dir, batch_size=8, shuffle=True):
    dataset = OutdoorDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)