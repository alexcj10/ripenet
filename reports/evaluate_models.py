import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Add project root to path so we can import from src directories
BASE_DIR = r"C:\Users\ALEX\Downloads\dataset"
sys.path.insert(0, BASE_DIR)

# IMPORT YOUR MODELS
from src_identity.model import FruitIdentityModel
from src.model import FruitRipenessModel
from src_regression.model import FruitRegressionModel
from src.multi_task_model import RipeNetMTL

# ==========================
# CONFIG
# ==========================

REPORT_DIR = os.path.join(BASE_DIR, "reports")
IMAGE_ROOT = os.path.join(REPORT_DIR, "image")
LABELS_CSV = os.path.join(REPORT_DIR, "labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

METRIC_DIR = os.path.join(REPORT_DIR, "metrics")
PLOT_DIR = os.path.join(REPORT_DIR, "plots")

os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common fruits available in both V1 and V2
COMMON_FRUITS = ["apple", "banana", "orange", "papaya"]

# Full class maps for each model
V1_FRUIT_MAP = ["apple", "banana", "orange", "papaya"]
V2_FRUIT_MAP = ['apple', 'banana', 'mango', 'orange', 'papaya', 'pineapple']
STAGE_MAP = ["unripe", "ripe", "rotten"]

# ==========================
# TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==========================
# NORMALIZATION
# ==========================

def normalize_days(x):
    if x <= 0:
        return abs(x) + 1.0
    return x

# ==========================
# LOAD LABELS & FILTER
# ==========================

df = pd.read_csv(LABELS_CSV)

# Keep only common fruits as requested
print(f"Initial labels: {len(df)}")
df = df[df["fruit"].isin(COMMON_FRUITS)].reset_index(drop=True)
print(f"Labels after filtering to common fruits {COMMON_FRUITS}: {len(df)}")

# ==========================
# LOAD V1 MODELS
# ==========================

print("Loading V1 models...")

v1_fruit = FruitIdentityModel()
v1_fruit.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_identity_model.pth"), map_location=DEVICE))
v1_fruit.to(DEVICE)
v1_fruit.eval()

v1_stage = FruitRipenessModel(num_classes=3)
v1_stage.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE))
v1_stage.to(DEVICE)
v1_stage.eval()

v1_time = FruitRegressionModel()
v1_time.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_regression_model.pth"), map_location=DEVICE))
v1_time.to(DEVICE)
v1_time.eval()

# ==========================
# LOAD V2 MODEL
# ==========================

print("Loading V2 model...")

v2_model = RipeNetMTL()
v2_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ripenet_v2_mtl.pth"), map_location=DEVICE))
v2_model.to(DEVICE)
v2_model.eval()

print("Models loaded successfully.\n")

# ==========================
# EVALUATION LOOP
# ==========================

results = []

for _, row in df.iterrows():

    img_path = os.path.join(IMAGE_ROOT, row["image_path"])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    # -------- V1 --------
    t0 = time.time()

    fruit_logits = v1_fruit(x)
    stage_logits = v1_stage(x)
    days_pred = v1_time(x)

    v1_time_taken = time.time() - t0

    v1_fruit_pred = V1_FRUIT_MAP[fruit_logits.argmax(dim=1).item()]
    v1_stage_pred = STAGE_MAP[stage_logits.argmax(dim=1).item()]
    v1_days_pred  = normalize_days(days_pred.item())

    # -------- V2 --------
    t0 = time.time()

    v2_out = v2_model(x)
    fruit_logits_v2 = v2_out["fruit"]
    stage_logits_v2 = v2_out["ripeness"]
    days_pred_v2 = v2_out["days"]

    v2_time_taken = time.time() - t0

    v2_fruit_pred = V2_FRUIT_MAP[fruit_logits_v2.argmax(dim=1).item()]
    v2_stage_pred = STAGE_MAP[stage_logits_v2.argmax(dim=1).item()]
    v2_days_pred  = normalize_days(days_pred_v2.squeeze().item())

    results.append({
        "fruit_gt": row["fruit"],
        "stage_gt": row["stage_name"],
        "days_gt": row["days"],
        "v1_fruit": v1_fruit_pred,
        "v1_stage": v1_stage_pred,
        "v1_days": v1_days_pred,
        "v2_fruit": v2_fruit_pred,
        "v2_stage": v2_stage_pred,
        "v2_days": v2_days_pred,
        "v1_time": v1_time_taken,
        "v2_time": v2_time_taken
    })

print("Inference completed.\n")

res = pd.DataFrame(results)

# ==========================
# METRICS
# ==========================

metrics = {
    "V1 Fruit Accuracy": accuracy_score(res["fruit_gt"], res["v1_fruit"]),
    "V2 Fruit Accuracy": accuracy_score(res["fruit_gt"], res["v2_fruit"]),
    "V1 Stage Accuracy": accuracy_score(res["stage_gt"], res["v1_stage"]),
    "V2 Stage Accuracy": accuracy_score(res["stage_gt"], res["v2_stage"]),
    "V1 MAE": mean_absolute_error(res["days_gt"], res["v1_days"]),
    "V2 MAE": mean_absolute_error(res["days_gt"], res["v2_days"]),
    "V1 RMSE": np.sqrt(mean_squared_error(res["days_gt"], res["v1_days"])),
    "V2 RMSE": np.sqrt(mean_squared_error(res["days_gt"], res["v2_days"])),
    "V1 Avg Time (ms)": res["v1_time"].mean() * 1000,
    "V2 Avg Time (ms)": res["v2_time"].mean() * 1000
}

metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_csv(os.path.join(METRIC_DIR, "summary.csv"), index=False)

print(metrics_df)
print("\n✅ Evaluation Complete")

# ==========================
# CLEAN RESEARCH-STYLE BENCHMARK PLOTS (MINIMAL)
# ==========================

print("\nGenerating clean research-style benchmark figures...")

import matplotlib as mpl

# ---- Clean Academic Style ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Default research-friendly colors
V1_COLOR = "#4C72B0"   # Muted Blue
V2_COLOR = "#DD8452"   # Muted Orange


def draw_bar(ax, title, v1, v2, ylabel):

    is_accuracy = "Accuracy" in title
    display_v1 = v1 * 100 if is_accuracy else v1
    display_v2 = v2 * 100 if is_accuracy else v2

    # Bars closer together
    bars = ax.bar(
        ["V1", "V2"],
        [display_v1, display_v2],
        color=[V1_COLOR, V2_COLOR],
        width=0.7   # wider bars = less space between them
    )

    ax.set_title(title, pad=15)
    ax.set_ylabel(ylabel)

    if is_accuracy:
        ax.set_ylim(0, 100)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.2f}%" if is_accuracy else f"{height:.2f}"
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + (2 if is_accuracy else height * 0.03),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold"
        )


# ==========================
# 1️⃣ COMBINED GRID FIGURE
# ==========================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
plt.subplots_adjust(hspace=0.5, wspace=0.25)

grid_metrics = [
    ("Fruit Classification Accuracy",
     metrics["V1 Fruit Accuracy"],
     metrics["V2 Fruit Accuracy"],
     "Accuracy (%)"),

    ("Ripeness Stage Accuracy",
     metrics["V1 Stage Accuracy"],
     metrics["V2 Stage Accuracy"],
     "Accuracy (%)"),

    ("Time Prediction Error (MAE)",
     metrics["V1 MAE"],
     metrics["V2 MAE"],
     "Days (Lower is Better)"),

    ("Inference Latency",
     metrics["V1 Avg Time (ms)"],
     metrics["V2 Avg Time (ms)"],
     "Latency (ms - Lower is Better)")
]

for ax, args in zip(axes.flatten(), grid_metrics):
    draw_bar(ax, *args)

plt.suptitle(
    "RipeNet V1 vs V2: Multi-Task Learning Benchmark",
    fontsize=15,
    weight="bold",
    y=0.97
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, "benchmark_comparison.png"))
plt.close()


# ==========================
# 2️⃣ INDIVIDUAL FIGURES
# ==========================

individual_plots = [
    ("Fruit Classification Accuracy",
     metrics["V1 Fruit Accuracy"],
     metrics["V2 Fruit Accuracy"],
     "Accuracy (%)",
     "fruit_accuracy.png"),

    ("Ripeness Stage Accuracy",
     metrics["V1 Stage Accuracy"],
     metrics["V2 Stage Accuracy"],
     "Accuracy (%)",
     "stage_accuracy.png"),

    ("Time Prediction Error (MAE)",
     metrics["V1 MAE"],
     metrics["V2 MAE"],
     "Days (Lower is Better)",
     "mae_comparison.png"),

    ("Time Prediction Error (RMSE)",
     metrics["V1 RMSE"],
     metrics["V2 RMSE"],
     "Days (Lower is Better)",
     "rmse_comparison.png"),

    ("Inference Latency",
     metrics["V1 Avg Time (ms)"],
     metrics["V2 Avg Time (ms)"],
     "Latency (ms)",
     "inference_time.png")
]

for title, v1, v2, ylabel, filename in individual_plots:
    fig, ax = plt.subplots(figsize=(6,5))
    draw_bar(ax, title, v1, v2, ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

print("✅ Clean benchmark figures generated successfully.")
