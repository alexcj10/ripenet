import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import FruitRegressionDataset
from model import FruitRegressionModel


# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

DATASET_DIR = ".."
IMAGES_DIR = os.path.join(DATASET_DIR, "train_selected")

TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")

MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "saved_models", "best_regression_model.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
# ------------------------


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    print(f"🚀 Training Regression Model on {DEVICE}...")

    # Datasets
    train_dataset = FruitRegressionDataset(TRAIN_CSV, IMAGES_DIR)
    val_dataset = FruitRegressionDataset(VAL_CSV, IMAGES_DIR)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = FruitRegressionModel().to(DEVICE)

    # Loss + Optimizer
    criterion = nn.MSELoss() # 🎯 Mean Squared Error for regression
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        # Print Mean Absolute Error (informational)
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f}"
        )

        # Save best model based on lowest loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best regression model saved (Val MSE: {val_loss:.4f})")

    print("\n🎉 Regression Training completed!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
