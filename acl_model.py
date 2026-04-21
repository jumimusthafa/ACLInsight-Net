# -*- coding: utf-8 -*-
"""
acl_model.py — ACLInsight-Net
EfficientNet-B0 backbone + single linear head, plus freeze / unfreeze helpers.
"""

import timm
import torch
import torch.nn as nn

from acl_config import device


# ── Model ─────────────────────────────────────────────────────────────────
class ACLModel(nn.Module):
    """
    EfficientNet-B0 backbone with a lightweight dropout + linear head.
    Single-neuron output (BCEWithLogitsLoss compatible).

    Design choice: one linear layer (1280 → 1) instead of a deep MLP.
    With ~1 000 training exams the simpler head avoids overfitting while
    the pretrained backbone already provides rich 1 280-d embeddings.
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,       # remove classifier head
            global_pool='avg'
        )
        embed_dim = self.backbone.num_features   # 1 280 for B0

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ── Freeze / Unfreeze Helpers ─────────────────────────────────────────────
def freeze_backbone(model: ACLModel, freeze: bool = True) -> None:
    """Freeze or unfreeze the entire EfficientNet backbone."""
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def unfreeze_last_n_blocks(model: ACLModel, n: int = 2) -> None:
    """
    Stage-2 fine-tuning: unfreeze only the last *n* MBConv block groups.
    Starts fully frozen, then re-enables the tail blocks to adapt
    pretrained features to the MRI domain without catastrophic forgetting.
    """
    freeze_backbone(model, freeze=True)          # start fully frozen
    blocks = list(model.backbone.blocks)         # list of MBConv groups
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True


# ── Quick Sanity Check ────────────────────────────────────────────────────
if __name__ == '__main__':
    model = ACLModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ACLModel: {n_params:,} total parameters")

    dummy = torch.randn(2, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # expected: (2, 1)
