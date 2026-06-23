import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from dataset import FruitRipenessDataset
from model import FruitRipenessModel


# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = ".."
IMAGES_DIR = os.path.join(DATASET_DIR, "train_selected")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

MODEL_PATH = os.path.join(DATASET_DIR, "saved_models", "best_model.pth")
BATCH_SIZE = 32
# ------------------------


def main():
    # Load dataset
    test_dataset = FruitRipenessDataset(TEST_CSV, IMAGES_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = FruitRipenessModel(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Test Accuracy: {acc:.4f}\n")

    print("📊 Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Unripe", "Fresh/Ripe", "Rotten"]
    ))


if __name__ == "__main__":
    main()
