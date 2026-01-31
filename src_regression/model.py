import torch
import torch.nn as nn
from torchvision import models


class FruitRegressionModel(nn.Module):
    def __init__(self):
        super(FruitRegressionModel, self).__init__()

        # Load pretrained EfficientNet
        # Weights instead of pretrained=True to avoid warnings in newer torchvision versions
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze backbone (transfer learning)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace classifier head for Regression (1 output)
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 🎯 Single value for "days_remaining"
        )

    def forward(self, x):
        return self.backbone(x)
