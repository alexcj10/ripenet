import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import FruitRegressionDataset
from model import FruitRegressionModel


# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = ".."
IMAGES_DIR = os.path.join(DATASET_DIR, "train_selected")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

MODEL_PATH = os.path.join(DATASET_DIR, "saved_models", "best_regression_model.pth")
BATCH_SIZE = 32
# ------------------------


def main():
    print("📝 Evaluating Regression Model...")

    # Load dataset
    test_dataset = FruitRegressionDataset(TEST_CSV, IMAGES_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = FruitRegressionModel()
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    print("-" * 30)
    print(f"🎯 Test MAE: {mae:.4f} days")
    print(f"🎯 Test RMSE: {rmse:.4f} days")
    print("-" * 30)
    
    # Sample predictions
    print("\n🔍 Sample Predictions (Target vs Prediction):")
    for i in range(min(10, len(all_preds))):
        print(f"Target: {all_targets[i]:.1f} days | Predicted: {all_preds[i]:.2f} days")


if __name__ == "__main__":
    main()
