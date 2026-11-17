"""PyTorch models (optional dependency).

* ``SmallCNN``       – compact from-scratch baseline CNN.
* ``resnet18_transfer`` – ImageNet-pretrained ResNet18 adapted to 1-channel input.

Install with ``pip install -r requirements-cnn.txt``.
"""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional
    raise ImportError("PyTorch is required for CNN models: pip install -r requirements-cnn.txt") from exc


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.features[-1]


def resnet18_transfer(n_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    w = m.conv1.weight.data.mean(dim=1, keepdim=True)          # RGB → grayscale stem
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, n_classes))
    m.gradcam_layer = m.layer4  # type: ignore[attr-defined]
    return m


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if not name.startswith(("fc", "head")):
            p.requires_grad = trainable


def build(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    return resnet18_transfer(n_classes, pretrained) if arch == "resnet18" else SmallCNN(n_classes)
