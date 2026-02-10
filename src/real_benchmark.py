import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
import time

# Local imports
from multi_task_model import RipeNetMTL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models"))
IMAGE_ROOT = "c:\\Users\\ALEX\\Downloads\\dataset\\RipeNet 2.0"

# --- V1 ARCHITECTURES (Extracted from old API) ---
class V1_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x): return self.backbone(x)

class V1_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.backbone(x)

# --- DATASET ---
class SimpleDataset(Dataset):
    def __init__(self, csv_file, root):
        self.df = pd.read_csv(csv_file)
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["relative_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, row["fruit_id"], row["ripeness_id"], row["days_remaining"]

def benchmark():
    print(f"🚀 Benchmarking on {DEVICE}...")
    
    # Load V2
    v2_model = RipeNetMTL().to(DEVICE)
    v2_model.load_state_dict(torch.load(os.path.join(SAVED_MODELS, "ripenet_v2_mtl.pth"), map_location=DEVICE))
    v2_model.eval()

    # Load V1 models
    v1_id = V1_Classifier(4).to(DEVICE)
    v1_id.load_state_dict(torch.load(os.path.join(SAVED_MODELS, "best_identity_model.pth"), map_location=DEVICE))
    v1_id.eval()

    v1_rip = V1_Classifier(3).to(DEVICE)
    v1_rip.load_state_dict(torch.load(os.path.join(SAVED_MODELS, "best_model.pth"), map_location=DEVICE))
    v1_rip.eval()

    v1_reg = V1_Regression().to(DEVICE)
    v1_reg.load_state_dict(torch.load(os.path.join(SAVED_MODELS, "best_regression_model.pth"), map_location=DEVICE))
    v1_reg.eval()

    # Data
    dataset = SimpleDataset("val_v2.csv", IMAGE_ROOT)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    v2_id_correct = 0
    v2_rip_correct = 0
    v2_reg_mae = 0

    v1_id_correct = 0
    v1_rip_correct = 0
    v1_reg_mae = 0
    v1_count = 0 # Only counts Apple, Banana, Orange, Papaya

    total = 0
    
    # Mappings
    # V1 Fruits: {0: apple, 1: banana, 2: orange, 3: papaya}
    # V2 Fruits: {0: apple, 1: banana, 2: mango, 3: orange, 4: papaya, 5: pineapple}
    v2_to_v1_fruit = {0: 0, 1: 1, 3: 2, 4: 3} 

    with torch.no_grad():
        for imgs, f_gt, r_gt, d_gt in loader:
            imgs, f_gt, r_gt, d_gt = imgs.to(DEVICE), f_gt.to(DEVICE), r_gt.to(DEVICE), d_gt.to(DEVICE)
            
            # V2 Predictions
            outs = v2_model(imgs)
            v2_id_correct += (outs["fruit"].argmax(1) == f_gt).sum().item()
            v2_rip_correct += (outs["ripeness"].argmax(1) == r_gt).sum().item()
            v2_reg_mae += torch.abs(outs["days"].squeeze() - d_gt).sum().item()

            # V1 Predictions (Only for compatible fruits)
            for i in range(len(f_gt)):
                fruit_idx = f_gt[i].item()
                if fruit_idx in v2_to_v1_fruit:
                    v1_f_gt = v2_to_v1_fruit[fruit_idx]
                    
                    # Single img pass for V1
                    v1_img = imgs[i].unsqueeze(0)
                    
                    v1_id_out = v1_id(v1_img)
                    v1_rip_out = v1_rip(v1_img)
                    v1_reg_out = v1_reg(v1_img)
                    
                    v1_id_correct += (v1_id_out.argmax(1).item() == v1_f_gt)
                    v1_rip_correct += (v1_rip_out.argmax(1).item() == r_gt[i].item())
                    v1_reg_mae += abs(v1_reg_out.item() - d_gt[i].item())
                    v1_count += 1
            
            total += len(f_gt)

    print("\n" + "="*40)
    print("📋 REAL BENCHMARK RESULTS")
    print("="*40)
    print(f"RipeNet V2 (All 6 Fruits):")
    print(f"  - Identity Accuracy: {v2_id_correct/total:.2%}")
    print(f"  - Ripeness Accuracy: {v2_rip_correct/total:.2%}")
    print(f"  - Regression MAE:    {v2_reg_mae/total:.2f} days")
    
    print(f"\nRipeNet V1 (Original 4 Fruits):")
    print(f"  - Identity Accuracy: {v1_id_correct/v1_count:.2%}")
    print(f"  - Ripeness Accuracy: {v1_rip_correct/v1_count:.2%}")
    print(f"  - Regression MAE:    {v1_reg_mae/v1_count:.2f} days")
    print("="*40)

if __name__ == "__main__":
    benchmark()
