import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import FruitRipenessDataset
from model import FruitRipenessModel


# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

DATASET_DIR = ".."
IMAGES_DIR = os.path.join(DATASET_DIR, "train_selected")

TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")

MODEL_SAVE_PATH = "../saved_models/best_model.pth"
os.makedirs("../saved_models", exist_ok=True)
# ------------------------


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return running_loss / len(loader), acc


def main():
    # Datasets
    train_dataset = FruitRipenessDataset(TRAIN_CSV, IMAGES_DIR)
    val_dataset = FruitRipenessDataset(VAL_CSV, IMAGES_DIR)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = FruitRipenessModel(num_classes=3).to(DEVICE)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Best model saved")

    print("🎉 Training completed!")


if __name__ == "__main__":
    main()
