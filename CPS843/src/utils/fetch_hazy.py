import os
import shutil
from pathlib import Path

SOURCE_SOTS_DIR = Path(r'C:\Users\knobu\Downloads\SOTS') 

CLEAR_DIR = Path('data/raw/train/clear')
HAZY_DIR = Path('data/raw/train/hazy')

def fetch_hazy_partners():
    if not HAZY_DIR.exists():
        os.makedirs(HAZY_DIR)

    clear_images = list(CLEAR_DIR.glob('*.png')) + list(CLEAR_DIR.glob('*.jpg'))
    
    print(f"Scanning {len(clear_images)} clear images to fetch their hazy partners...")
    
    matched_count = 0
    removed_count = 0
    
    for clear_path in clear_images:
        img_id = clear_path.stem 
        
        candidates = list(SOURCE_SOTS_DIR.glob(f"{img_id}_*0.2.png")) + \
                     list(SOURCE_SOTS_DIR.glob(f"{img_id}_*0.2.jpg"))
        
        if candidates:
            best_match = candidates[0]
            
            shutil.copy2(best_match, HAZY_DIR / best_match.name)
            
            matched_count += 1
            if matched_count % 100 == 0:
                print(f"Fetched {matched_count} hazy images...")
        
        else:
            os.remove(clear_path)
            removed_count += 1

    print("------------------------------------------------")
    print(f"Complete.")
    print(f"Matched Pairs Created: {matched_count}")
    print(f"Orphan Clear Images Removed: {removed_count}")
    
    num_clear = len(list(CLEAR_DIR.glob('*')))
    num_hazy = len(list(HAZY_DIR.glob('*')))
    print(f"Final Count -> Clear: {num_clear} | Hazy: {num_hazy}")
    
    if num_clear == num_hazy and num_clear > 0:
        print("SUCCESS: Folders are synced. Ready to train.")
    else:
        print("WARNING: Counts do not match.")

if __name__ == "__main__":
    fetch_hazy_partners()