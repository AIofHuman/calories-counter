# src/multimodal.py
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ImageEncoderEffNet(nn.Module):
    def __init__(self, name: str = "efficientnet_b0", pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        if not hasattr(tvm, name):
            raise ValueError(f"Unknown torchvision model: {name}")

        weights = "DEFAULT" if pretrained else None
        m = getattr(tvm, name)(weights=weights)

        # EfficientNet classifier is typically: Sequential(Dropout, Linear)
        in_dim = m.classifier[-1].in_features
        m.classifier = nn.Identity()

        self.backbone = m
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.proj(feat)


class TextMLP(nn.Module):
    def __init__(self, n_ingr: int = 552, hidden: int = 256, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ingr, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalRegressor(nn.Module):
    def __init__(
        self,
        n_ingr: int = 555,
        image_model: str = "efficientnet_b0",
        image_pretrained: bool = True,
        img_dim: int = 512,
        txt_hidden: int = 256,
        txt_dim: int = 128,
        head_hidden: int = 512,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.img = ImageEncoderEffNet(name=image_model, pretrained=image_pretrained, out_dim=img_dim)
        self.txt = TextMLP(n_ingr=n_ingr, hidden=txt_hidden, out_dim=txt_dim, dropout=head_dropout)

        self.head = nn.Sequential(
            nn.Linear(img_dim + txt_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, image: torch.Tensor, ingr_multihot: torch.Tensor) -> torch.Tensor:
        img_feat = self.img(image)
        txt_feat = self.txt(ingr_multihot)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        y = self.head(fused).squeeze(1)
        return y
