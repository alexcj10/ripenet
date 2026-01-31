import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FruitRegressionDataset(Dataset):
    def __init__(self, csv_file, images_root):
        """
        csv_file: path to train.csv / val.csv / test.csv
        images_root: path to train_selected folder
        """
        self.data = pd.read_csv(csv_file)
        self.images_root = images_root

        # Image preprocessing (ImageNet standard)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_name = row["image_name"]
        fruit = row["fruit"]
        stage = row["stage"]
        days = float(row["days_remaining"]) # 🎯 Target for regression

        # 📁 Handle inconsistent folder naming (spaced/unspaced, singular/plural)
        potential_folders = [
            f"{stage} {fruit}",
            f"{stage} {fruit}s",
            f"{stage}{fruit}",
            f"{stage}{fruit}s"
        ]
        
        image_path = None
        for folder in potential_folders:
            test_path = os.path.join(self.images_root, folder, image_name)
            if os.path.exists(test_path):
                image_path = test_path
                break

        if image_path is None:
            checked = ", ".join(potential_folders)
            raise FileNotFoundError(f"❌ Could not find {image_name} in any of: {checked}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # 🔑 Return days_remaining as a float tensor for regression
        target = torch.tensor([days], dtype=torch.float32)

        return image, target
