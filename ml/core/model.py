import torch
import torch.nn as nn
from torchvision import models


class FruitRipenessModel(nn.Module):
    def __init__(self, num_classes=3):
        super(FruitRipenessModel, self).__init__()

        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Freeze convolution layers (transfer learning)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
