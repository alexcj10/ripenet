import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("labels_selected.csv")

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["stage"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["stage"]
)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✅ train.csv, val.csv, test.csv created")
