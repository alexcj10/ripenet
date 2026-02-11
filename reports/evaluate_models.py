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
# FINAL RESEARCH-LEVEL BENCHMARK PLOTS (NO OVERLAP)
# ==========================

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import numpy as np

print("\nGenerating final research-level benchmark plots...")

# ---- STYLE CONFIG ----
sns.set_theme(style="whitegrid", context="paper")

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

V1_COLOR = "#94A3B8"   # Slate
V2_COLOR = "#6366F1"   # Indigo
DELTA_COLOR = "#10B981"  # Emerald green

# ---- METRIC DEFINITIONS ----
metric_info = [
    ("Fruit Classification Accuracy",
     metrics["V1 Fruit Accuracy"] * 100,
     metrics["V2 Fruit Accuracy"] * 100,
     True,
     "Accuracy (%)"),

    ("Ripeness Stage Accuracy",
     metrics["V1 Stage Accuracy"] * 100,
     metrics["V2 Stage Accuracy"] * 100,
     True,
     "Accuracy (%)"),

    ("Time Prediction Error (MAE)",
     metrics["V1 MAE"],
     metrics["V2 MAE"],
     False,
     "Days (Lower is Better)"),

    ("Inference Latency",
     metrics["V1 Avg Time (ms)"],
     metrics["V2 Avg Time (ms)"],
     False,
     "Latency (ms - Lower is Better)")
]

# ==========================
# 2x2 GRID FIGURE
# ==========================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for ax, (title, v1, v2, higher_is_better, ylabel) in zip(axes.flatten(), metric_info):

    data = pd.DataFrame({
        "Model": ["V1 (Baseline)", "V2 (RipeNet)"],
        "Value": [v1, v2]
    })

    sns.barplot(
        data=data,
        x="Model",
        y="Value",
        palette=[V1_COLOR, V2_COLOR],
        ax=ax
    )

    ax.set_title(title, pad=15)
    ax.set_ylabel(ylabel)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    # Compute improvement
    if higher_is_better:
        improvement = ((v2 - v1) / v1) * 100
        delta_text = f"+{improvement:.1f}% Improvement"
    else:
        improvement = ((v1 - v2) / v1) * 100
        delta_text = f"-{improvement:.1f}% Faster/Reduced"

    # Place improvement text INSIDE plot safely
    ax.text(
        0.5,
        0.90,
        delta_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=DELTA_COLOR,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3)
    )

fig.suptitle(
    "RipeNet V1 vs V2: Multi-Task Learning Benchmark",
    fontsize=15,
    weight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(PLOT_DIR, "benchmark_comparison.png"))
plt.close()

print("✅ Clean benchmark grid saved.")

# ==========================
# INDIVIDUAL METRIC FIGURES
# ==========================

for title, v1, v2, higher_is_better, ylabel in metric_info:

    plt.figure(figsize=(6,5))

    data = pd.DataFrame({
        "Model": ["V1 (Baseline)", "V2 (RipeNet)"],
        "Value": [v1, v2]
    })

    ax = sns.barplot(
        data=data,
        x="Model",
        y="Value",
        palette=[V1_COLOR, V2_COLOR]
    )

    plt.title(title, pad=15)
    plt.ylabel(ylabel)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    if higher_is_better:
        improvement = ((v2 - v1) / v1) * 100
        delta_text = f"+{improvement:.1f}% Improvement"
    else:
        improvement = ((v1 - v2) / v1) * 100
        delta_text = f"-{improvement:.1f}% Faster/Reduced"

    plt.text(
        0.5,
        0.88,
        delta_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=DELTA_COLOR,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3)
    )

    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

print("✅ Individual benchmark figures saved.")
