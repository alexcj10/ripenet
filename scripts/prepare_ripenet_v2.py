import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Configuration
DATASET_DIR = "RipeNet 2.0"
OUTPUT_CSV = "ripenet_v2_master.csv"
TRAIN_CSV = "train_v2.csv"
VAL_CSV = "val_v2.csv"

# Mapping
FRUIT_MAP = {
    "apple": 0,
    "banana": 1,
    "mango": 2,
    "orange": 3,
    "papaya": 4,
    "pineapple": 5
}

STAGE_MAP = {
    "Unripe": 0,
    "Fresh": 1,
    "Rotten": 2
}

# Professional Timeline Mapping (Agreed)
TIMELINES = {
    "apple": {"Unripe": 10.0, "Fresh": 5.0, "Rotten": 0.0},
    "banana": {"Unripe": 6.0, "Fresh": 3.0, "Rotten": 0.0},
    "mango": {"Unripe": 7.0, "Fresh": 3.0, "Rotten": 0.0},
    "orange": {"Unripe": 8.0, "Fresh": 5.0, "Rotten": 0.0},
    "papaya": {"Unripe": 5.0, "Fresh": 2.0, "Rotten": 0.0},
    "pineapple": {"Unripe": 6.0, "Fresh": 3.0, "Rotten": 0.0}
}

# Regression logic with Fruit-Specific Base Values
def get_regression_label(fruit, stage, filename):
    base = TIMELINES[fruit][stage]
    
    if stage == "Unripe":
        return base + random.uniform(-0.5, 0.5)
    elif stage == "Fresh":
        return base + random.uniform(-0.3, 0.3)
    elif stage == "Rotten":
        lower_name = filename.lower()
        # User requested range between -1 and -3 for very spoiled/black fruit
        if any(word in lower_name for word in ["black", "mold", "fungus", "dark", "spoiled_old"]):
            return random.uniform(-3.0, -1.0)
        
        # 20% chance of being "Old Rotten" if no keywords (to handle variation)
        if random.random() < 0.2:
            return random.uniform(-2.5, -1.0)
            
        # Starting rot or standard rotten (0.0 to 0.8)
        return random.uniform(0.0, 0.8)

    return 0.0

def main():
    data = []
    
    # Walk through the directory structure
    for fruit in FRUIT_MAP.keys():
        fruit_dir = os.path.join(DATASET_DIR, fruit)
        if not os.path.exists(fruit_dir):
            print(f"⚠️ Warning: {fruit_dir} not found.")
            continue
            
        for stage in STAGE_MAP.keys():
            stage_dir = os.path.join(fruit_dir, stage)
            if not os.path.exists(stage_dir):
                print(f"⚠️ Warning: {stage_dir} not found.")
                continue
                
            files = [f for f in os.listdir(stage_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"🔍 Found {len(files)} images for {fruit} -> {stage}")
            
            for f in files:
                reg_label = get_regression_label(fruit, stage, f)
                data.append({
                    "image_name": f,
                    "fruit": fruit,
                    "stage": stage.lower(),
                    "fruit_id": FRUIT_MAP[fruit],
                    "ripeness_id": STAGE_MAP[stage],
                    "days_remaining": round(reg_label, 2),
                    "relative_path": os.path.join(fruit, stage, f)
                })

    df = pd.DataFrame(data)
    
    if df.empty:
        print("❌ No images found! Check your RipeNet 2.0 folder path.")
        return

    # Save master
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Generated {OUTPUT_CSV} with {len(df)} rows.")

    # Train/Val Split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[['fruit_id', 'ripeness_id']], random_state=42)
    
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    
    print(f"📂 Split data saved to {TRAIN_CSV} and {VAL_CSV}")

if __name__ == "__main__":
    main()
