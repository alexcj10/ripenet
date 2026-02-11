import os
import csv

# CHANGE PATH IF NEEDED
IMAGE_ROOT = r"C:\Users\ALEX\Downloads\dataset\reports\image"
OUTPUT_CSV = r"C:\Users\ALEX\Downloads\dataset\reports\labels.csv"

# Stage labels
STAGE_LABELS = {
    "unripe": 0,
    "ripe": 1,
    "rotten": 2
}

# Fruit-wise, stage-wise day mapping (FAIR & FIXED)
DAY_MAPPING = {
    "apple":   {"unripe": 10, "ripe": 5, "rotten": 2},
    "banana":  {"unripe": 6,  "ripe": 3, "rotten": 1},
    "orange":  {"unripe": 8,  "ripe": 4, "rotten": 2},
    "papaya":  {"unripe": 6,  "ripe": 3, "rotten": 1},
}

rows = []

for fruit in DAY_MAPPING.keys():
    fruit_path = os.path.join(IMAGE_ROOT, fruit)
    if not os.path.isdir(fruit_path):
        continue

    for stage in ["unripe", "ripe", "rotten"]:
        stage_path = os.path.join(fruit_path, stage)
        if not os.path.isdir(stage_path):
            continue

        for img in os.listdir(stage_path):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            rows.append([
                os.path.join(fruit, stage, img),
                fruit,
                STAGE_LABELS[stage],
                stage,
                DAY_MAPPING[fruit][stage]
            ])

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_path",
        "fruit",
        "stage_label",
        "stage_name",
        "days"
    ])
    writer.writerows(rows)

print(f"✅ labels.csv generated successfully with {len(rows)} samples")
