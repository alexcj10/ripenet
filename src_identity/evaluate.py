import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from dataset import FruitIdentityDataset
from model import FruitIdentityModel

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = ".."
IMAGES_DIR = os.path.join(DATASET_DIR, "train_selected")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
MODEL_PATH = os.path.join(DATASET_DIR, "saved_models", "best_identity_model.pth")
BATCH_SIZE = 32
# ------------------------

def main():
    print("📊 Evaluating Fruit Identity Model...")
    
    test_dataset = FruitIdentityDataset(TEST_CSV, IMAGES_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = FruitIdentityModel()
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Identity Accuracy: {acc:.4f}\n")
    
    print("📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Apple", "Banana", "Orange", "Papaya"]))

if __name__ == "__main__":
    main()
