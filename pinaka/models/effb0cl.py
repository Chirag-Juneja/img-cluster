import torch
import torch.nn as nn
import torchvision.models as models
import copy


class EffB0CL(nn.Module):
    def __init__(self, classes=[], features=None, dropout: float = 0.2):
        super().__init__()

        if not features:
            features = models.efficientnet_b0(weights="IMAGENET1K_V1").features

        self.features = copy.deepcopy(features)

        self.classes = classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=1280, out_features=len(self.classes)),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
