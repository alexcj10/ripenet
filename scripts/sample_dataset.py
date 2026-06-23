import os
import random
import shutil

random.seed(42)

SRC_DIR = "train"
DST_DIR = "train_selected"

TARGET_COUNTS = {
    "unripe apple": 600,
    "freshapples": 600,
    "rottenapples": 600,

    "unripe banana": 600,
    "freshbanana": 600,
    "rottenbanana": 600,

    "unripe orange": 600,
    "freshoranges": 600,
    "rottenoranges": 600,

    "unripe papaya": 400,
    "ripe papaya": 400,
}

os.makedirs(DST_DIR, exist_ok=True)

for folder, count in TARGET_COUNTS.items():
    src_path = os.path.join(SRC_DIR, folder)
    dst_path = os.path.join(DST_DIR, folder)

    os.makedirs(dst_path, exist_ok=True)

    files = [f for f in os.listdir(src_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    selected = random.sample(files, min(count, len(files)))

    for f in selected:
        shutil.copy(os.path.join(src_path, f), dst_path)

    print(f"✅ {folder}: {len(selected)} images copied")

print("🎉 Dataset sampling completed")
