import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def clean_dataset(csv_path, images_root):
    print(f"🧹 Cleaning dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    valid_rows = []
    corrupt_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(images_root, row["relative_path"])
        
        try:
            if not os.path.exists(image_path):
                corrupt_count += 1
                continue
                
            # Try to actually open and verify the image
            with Image.open(image_path) as img:
                img.verify() # Verify file integrity
            
            # Re-open for actual conversion test (verify doesn't catch everything)
            with Image.open(image_path) as img:
                img.convert("RGB")
                
            valid_rows.append(row)
        except Exception:
            # print(f"❌ Corrupt file found: {image_path}")
            corrupt_count += 1

    new_df = pd.DataFrame(valid_rows)
    new_df.to_csv(csv_path, index=False)
    print(f"✅ Cleaned {csv_path}. Removed {corrupt_count} corrupt images. Remaining: {len(new_df)}")

if __name__ == "__main__":
    clean_dataset("train_v2.csv", "RipeNet 2.0")
    clean_dataset("val_v2.csv", "RipeNet 2.0")
