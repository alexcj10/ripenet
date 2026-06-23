import os
import csv

DATASET_DIR = "train_selected"
OUTPUT_CSV = "labels_selected.csv"

# folder_name : (fruit, stage, days_remaining)
LABEL_MAP = {
    # Apple
    "unripe apple":   ("apple", "unripe", 10),
    "freshapples":    ("apple", "fresh", 5),
    "rottenapples":   ("apple", "rotten", 2),

    # Banana
    "unripe banana":  ("banana", "unripe", 6),
    "freshbanana":    ("banana", "fresh", 3),
    "rottenbanana":   ("banana", "rotten", 1),

    # Orange
    "unripe orange":  ("orange", "unripe", 8),
    "freshoranges":   ("orange", "fresh", 4),
    "rottenoranges":  ("orange", "rotten", 2),

    # Papaya (2-stage)
    "unripe papaya":  ("papaya", "unripe", 6),
    "ripe papaya":    ("papaya", "ripe", 3),
}

rows = []

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)

    if folder not in LABEL_MAP:
        print(f"⚠️ Skipping unknown folder: {folder}")
        continue

    fruit, stage, days = LABEL_MAP[folder]

    for img in os.listdir(folder_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append([
                img,
                fruit,
                stage,
                days
            ])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "fruit", "stage", "days_remaining"])
    writer.writerows(rows)

print(f"✅ labels.csv created successfully with {len(rows)} images")
