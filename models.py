"""
İki sınıflandırıcı: ConvNeXt (transfer learning) ve özel CNN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

import config


def build_convnext_tiny(num_classes: int = config.NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Torchvision ConvNeXt-Tiny; son lineer katman ikili sınıfa uyarlanır."""
    weights = None
    if pretrained:
        w_enum = getattr(tvm, "ConvNeXt_Tiny_Weights", None)
        weights = w_enum.DEFAULT if w_enum is not None else "DEFAULT"
    model = tvm.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


class CustomHeadCTCNN(nn.Module):
    """
    Özel CNN: Conv + BN + ReLU + Pool tekrarı, ardından Dropout ve tam bağlı katman.
    Forward pass açıkça `def forward` içinde tanımlıdır.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri besleme: x [B,3,H,W] -> logits [B,num_classes].
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_model(model_name: str, num_classes: int = config.NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    name = model_name.lower().strip()
    if name in ("convnext", "convnext_tiny", "convnext-tiny"):
        return build_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
    if name in ("custom", "custom_cnn", "headctcnn"):
        return CustomHeadCTCNN(num_classes=num_classes)
    raise ValueError(f"Bilinmeyen model: {model_name}")
