import pandas as pd

# Load selected labels
df = pd.read_csv("labels_selected.csv")

def assign_days(row):
    fruit = row["fruit"].lower()
    stage = row["stage"].lower()

    if fruit == "papaya":
        if stage == "unripe":
            return 6
        elif stage == "ripe":
            return 2
    else:
        if stage == "unripe":
            return 7
        elif stage in ["fresh", "ripe"]:
            return 3
        elif stage == "rotten":
            return 0

    return None  # fallback

df["days_remaining"] = df.apply(assign_days, axis=1)

# Save updated labels
df.to_csv("labels_selected_regression.csv", index=False)

print("✅ labels_selected_regression.csv created successfully")
