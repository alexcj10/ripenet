import torch
from src.dataset import MultiTaskFruitDataset
from src.multi_task_model import RipeNetMTL
from torch.utils.data import DataLoader

def test_pipeline():
    print("📋 Starting Pipeline Verification...")
    
    # 1. Dataset Test
    try:
        ds = MultiTaskFruitDataset("train_v2.csv", "RipeNet 2.0", is_train=True)
        img, targets = ds[0]
        print(f"✅ Dataset: Image shape {img.shape}")
        print(f"✅ Dataset: Targets: {targets.keys()}")
        print(f"   - Fruit ID: {targets['fruit_id'].item()}")
        print(f"   - Ripeness ID: {targets['ripeness_id'].item()}")
        print(f"   - Days Remaining: {targets['days_remaining'].item()}")
    except Exception as e:
        print(f"❌ Dataset Failure: {e}")
        return

    # 2. Model Test
    try:
        model = RipeNetMTL()
        batch = img.unsqueeze(0) # Simulate batch size 1
        outputs = model(batch)
        print(f"✅ Model: Output keys {outputs.keys()}")
        print(f"   - Fruit Logits: {outputs['fruit'].shape}")
        print(f"   - Ripeness Logits: {outputs['ripeness'].shape}")
        print(f"   - Days Pred: {outputs['days'].shape}")
    except Exception as e:
        print(f"❌ Model Failure: {e}")
        return

    print("\n🔥 PIPELINE VERIFIED! You can now run 'python src/train.py' for full training.")

if __name__ == "__main__":
    test_pipeline()
