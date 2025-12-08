import os
from pathlib import Path

def check_alignment():
    clear_dir = Path('data/raw/train/clear')
    hazy_dir = Path('data/raw/train/hazy')

    clear_files = sorted(list(clear_dir.glob('*.png')) + list(clear_dir.glob('*.jpg')))
    hazy_files = sorted(list(hazy_dir.glob('*.png')) + list(hazy_dir.glob('*.jpg')))

    print(f"Found {len(clear_files)} Clear images.")
    print(f"Found {len(hazy_files)} Hazy images.")

    if len(clear_files) != len(hazy_files):
        print("!!! CRITICAL ERROR !!!")
        print("File counts do not match. You cannot train.")
        return

    print("\n--- Inspecting Pairs (Do they match?) ---")
    for i in [0, 1, 2, -1]: 
        c_name = clear_files[i].name
        h_name = hazy_files[i].name
        
        c_id = c_name.split('.')[0]      
        h_id = h_name.split('_')[0]      
        
        match_status = "MATCH " if c_id == h_id else "MISMATCH "
        print(f"Pair {i}: {c_name}  <-->  {h_name}  [{match_status}]")

if __name__ == "__main__":
    check_alignment()