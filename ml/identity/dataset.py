import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FruitIdentityDataset(Dataset):
    def __init__(self, csv_file, images_root):
        """
        csv_file: path to train.csv / val.csv / test.csv
        images_root: path to train_selected folder
        """
        self.data = pd.read_csv(csv_file)
        self.images_root = images_root
        
        # 0: apple, 1: banana, 2: orange, 3: papaya
        self.fruit_map = {"apple": 0, "banana": 1, "orange": 2, "papaya": 3}

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["image_name"]
        fruit = row["fruit"].lower()
        stage = row["stage"]
        
        # 🎯 Target: fruit species index
        label = self.fruit_map[fruit]

        # 📁 Folders are inconsistent (same as before)
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
            raise FileNotFoundError(f"❌ Image not found: {image_name}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return image, label
