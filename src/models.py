import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class Lens(nn.Module):
    def __init__(
        self,
        resnet_size = 18,
        n_classes   = 7,
        dropout     = 0.5,
        n_unfreeze  = 0,
        n_linear    = 1,
    ):
        super().__init__()

        if resnet_size == 18:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif resnet_size == 34:
            self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        for param in self.resnet.parameters():
            param.requires_grad = False

        layers = [self.resnet.layer1, self.resnet.layer2,
                  self.resnet.layer3, self.resnet.layer4]
        for layer in layers[-n_unfreeze:] if n_unfreeze > 0 else []:
            for param in layer.parameters():
                param.requires_grad = True

        self.resnet.fc = nn.Identity()

        if n_linear == 1:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(512, n_classes)
            )
        elif n_linear == 2:
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, image, meta=None):
        return self.fc(self.resnet(image))