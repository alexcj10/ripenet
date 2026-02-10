import torch
import torch.nn as nn
from torchvision import models

class RipeNetMTL(nn.Module):
    def __init__(self, num_fruits=6, num_ripeness=3):
        super(RipeNetMTL, self).__init__()
        
        # Load pre-trained EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Features from the backbone (1280 for EfficientNet-B0)
        self.in_features = self.backbone.classifier[1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # 🟢 Task 1: Fruit Identity Head (apple, banana, mango, orange, papaya, pineapple)
        self.identity_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_fruits)
        )
        
        # 🟡 Task 2: Ripeness Classification Head (Unripe, Fresh, Rotten)
        self.ripeness_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_ripeness)
        )
        
        # 🔴 Task 3: Regression Head (Days Remaining)
        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Shared feature extraction
        features = self.backbone(x)
        
        # Branch out for each task
        fruit_logits = self.identity_head(features)
        ripeness_logits = self.ripeness_head(features)
        days_remaining = self.regression_head(features)
        
        return {
            "fruit": fruit_logits,
            "ripeness": ripeness_logits,
            "days": days_remaining
        }

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
