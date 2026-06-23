import torch.nn as nn
from torchvision import models

class FruitIdentityModel(nn.Module):
    def __init__(self, num_classes=4):
        super(FruitIdentityModel, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Replace classifier for 4 fruit types
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
