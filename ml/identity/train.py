import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import FruitIdentityDataset
from model import FruitIdentityModel

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

DATASET_DIR = ".."
IMAGES_DIR = os.path.join(DATASET_DIR, "train_selected")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")

MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "saved_models", "best_identity_model.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
# ------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss/len(loader), correct/total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss/len(loader), correct/total

def main():
    print(f"🚀 Training Fruit Identity Model on {DEVICE}...")
    
    train_dataset = FruitIdentityDataset(TRAIN_CSV, IMAGES_DIR)
    val_dataset = FruitIdentityDataset(VAL_CSV, IMAGES_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = FruitIdentityModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best identity model saved (Acc: {val_acc:.4f})")

    print(f"\n🎉 Identity training complete! Saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
