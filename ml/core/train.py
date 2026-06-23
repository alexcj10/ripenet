import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Local imports
from dataset import MultiTaskFruitDataset
from multi_task_model import RipeNetMTL

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
PHASE1_EPOCHS = 5   # Just the heads
PHASE2_EPOCHS = 15  # Full fine-tuning
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5

TRAIN_CSV = "train_v2.csv"
VAL_CSV = "val_v2.csv"
IMAGES_ROOT = "RipeNet 2.0"

MODEL_SAVE_PATH = "saved_models/ripenet_v2_mtl.pth"
os.makedirs("saved_models", exist_ok=True)
# ------------------------

def train_one_epoch(model, loader, criterion_cls, criterion_reg, optimizer):
    model.train()
    running_total_loss = 0.0
    
    for images, targets in loader:
        images = images.to(DEVICE)
        fruit_labels = targets["fruit_id"].to(DEVICE)
        ripeness_labels = targets["ripeness_id"].to(DEVICE)
        days_labels = targets["days_remaining"].to(DEVICE)

        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Calculate individual losses
        loss_fruit = criterion_cls(outputs["fruit"], fruit_labels)
        loss_ripeness = criterion_cls(outputs["ripeness"], ripeness_labels)
        loss_days = criterion_reg(outputs["days"], days_labels)
        
        # Weighted Total Loss (1.0 each for classification, 0.1 for regression)
        total_loss = loss_fruit + loss_ripeness + 0.1 * loss_days
        
        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item()

    return running_total_loss / len(loader)

def validate(model, loader, criterion_cls, criterion_reg):
    model.eval()
    correct_fruit = 0
    correct_ripeness = 0
    total = 0
    running_reg_error = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            fruit_labels = targets["fruit_id"].to(DEVICE)
            ripeness_labels = targets["ripeness_id"].to(DEVICE)
            days_labels = targets["days_remaining"].to(DEVICE)

            outputs = model(images)
            
            # Identity accuracy
            fruit_preds = outputs["fruit"].argmax(dim=1)
            correct_fruit += (fruit_preds == fruit_labels).sum().item()
            
            # Ripeness accuracy
            ripeness_preds = outputs["ripeness"].argmax(dim=1)
            correct_ripeness += (ripeness_preds == ripeness_labels).sum().item()
            
            # Regression error (Mean Absolute Error)
            running_reg_error += torch.abs(outputs["days"] - days_labels).sum().item()
            
            total += fruit_labels.size(0)

    fruit_acc = correct_fruit / total
    ripeness_acc = correct_ripeness / total
    avg_mae = running_reg_error / total
    
    return fruit_acc, ripeness_acc, avg_mae

def main():
    # Datasets
    train_dataset = MultiTaskFruitDataset(TRAIN_CSV, IMAGES_ROOT, is_train=True)
    val_dataset = MultiTaskFruitDataset(VAL_CSV, IMAGES_ROOT, is_train=False)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = RipeNetMTL().to(DEVICE)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss() # Robust to outliers like -2 labels

    # PHASE 1: Train heads only
    print("\n🚀 PHASE 1: Training Heads Only (Frozen Backbone)")
    model.freeze_backbone()
    optimizer = AdamW(model.parameters(), lr=LR_PHASE1)
    
    for epoch in range(PHASE1_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion_cls, criterion_reg, optimizer)
        f_acc, r_acc, mae = validate(model, val_loader, criterion_cls, criterion_reg)
        print(f"Epoch [{epoch+1}/{PHASE1_EPOCHS}] Loss: {train_loss:.4f} | Fruit Acc: {f_acc:.4f} | Ripeness Acc: {r_acc:.4f} | Days MAE: {mae:.4f}")

    # PHASE 2: Full fine-tuning
    print("\n🔥 PHASE 2: Full Fine-tuning (Unfrozen Backbone)")
    model.unfreeze_backbone()
    optimizer = AdamW(model.parameters(), lr=LR_PHASE2)
    
    best_total_acc = 0.0
    
    for epoch in range(PHASE2_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion_cls, criterion_reg, optimizer)
        f_acc, r_acc, mae = validate(model, val_loader, criterion_cls, criterion_reg)
        
        # Save best combined accuracy
        combined_acc = (f_acc + r_acc) / 2
        print(f"Epoch [{epoch+1}/{PHASE2_EPOCHS}] Loss: {train_loss:.4f} | Fruit Acc: {f_acc:.4f} | Ripeness Acc: {r_acc:.4f} | Days MAE: {mae:.4f}")
        
        if combined_acc > best_total_acc:
            best_total_acc = combined_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Best professional model saved!")

    print("\n🎉 Training completed! RipeNet 2.0 is ready for production.")

if __name__ == "__main__":
    main()
