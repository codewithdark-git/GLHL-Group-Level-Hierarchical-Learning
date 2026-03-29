"""
glhl/models.py
==============
Backbone factory and group-specific classification head.

Supported backbones (all pretrained on ImageNet):
  - mobilenet_v2
  - resnet50
  - efficientnet_b0
  - efficientnet_b4

Each backbone is wrapped in a GroupModel that:
  1. Extracts a feature vector from the backbone.
  2. Applies dropout regularization.
  3. Projects to group-specific logits via a linear head.
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def build_backbone(backbone_name: str):
    """
    Load an ImageNet-pretrained backbone and return:
      (backbone_module, feature_dim)

    The returned backbone_module outputs a 1-D feature vector per sample
    (global average pooling already applied).
    """

    if backbone_name == "mobilenet_v2":
        base   = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classifier head; keep features only
        feat_dim = base.last_channel   # 1280
        backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    elif backbone_name == "resnet50":
        base     = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = base.fc.in_features   # 2048
        backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())

    elif backbone_name == "efficientnet_b0":
        base     = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = base.classifier[1].in_features   # 1280
        backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    elif backbone_name == "efficientnet_b4":
        base     = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        feat_dim = base.classifier[1].in_features   # 1792
        backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}. "
                         f"Choose from: mobilenet_v2, resnet50, efficientnet_b0, efficientnet_b4")

    return backbone, feat_dim


# ---------------------------------------------------------------------------
# GroupModel — backbone + dropout + linear head for one group
# ---------------------------------------------------------------------------

class GroupModel(nn.Module):
    """
    A group-specific classifier consisting of:
      1. A shared pretrained backbone (feature extractor).
      2. A dropout layer for regularization.
      3. A linear classification head sized to the group's class count.

    Parameters
    ----------
    backbone_name  : str   – one of the supported backbone names.
    num_classes    : int   – number of classes in this group (local label space).
    dropout        : float – dropout probability before the linear head.
    freeze_layers  : int   – number of backbone layers to freeze initially.
                             Set to 0 to train all layers from the start.
    """

    def __init__(self, backbone_name, num_classes, dropout=0.5, freeze_layers=50):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes   = num_classes

        self.backbone, self.feat_dim = build_backbone(backbone_name)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.classifier    = nn.Linear(self.feat_dim, num_classes)

        # Track freezing state
        self._freeze_layers = freeze_layers
        self._frozen        = False

    def freeze_backbone(self):
        """Freeze the first N backbone parameters for warm-up training."""
        params = list(self.backbone.parameters())
        n      = min(self._freeze_layers, len(params))
        for p in params[:n]:
            p.requires_grad = False
        self._frozen = True

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, x):
        features = self.backbone(x)          # (B, feat_dim)
        features = self.dropout_layer(features)
        logits   = self.classifier(features) # (B, num_classes)
        return logits

    def get_features(self, x):
        """Return feature vector before the classification head."""
        with torch.no_grad():
            features = self.backbone(x)
        return features


# ---------------------------------------------------------------------------
# Loss with optional class weighting and label smoothing
# ---------------------------------------------------------------------------

def build_loss(class_counts_for_group, label_smoothing=0.1, device="cpu"):
    """
    Build a weighted cross-entropy loss for a group.

    Class weights are computed as: w_c = N / (C * n_c)
    where N = total samples in group, C = num classes in group, n_c = count of class c.

    Parameters
    ----------
    class_counts_for_group : list[int]  – sample count per local class index.
    label_smoothing        : float      – label smoothing epsilon.
    device                 : str        – 'cuda' or 'cpu'.

    Returns
    -------
    nn.CrossEntropyLoss with class weights and label smoothing.
    """
    import torch

    counts = torch.tensor(class_counts_for_group, dtype=torch.float32)
    N      = counts.sum()
    C      = len(counts)
    weights = N / (C * counts)
    weights = weights / weights.sum() * C   # normalize so mean weight ≈ 1

    return nn.CrossEntropyLoss(
        weight          = weights.to(device),
        label_smoothing = label_smoothing
    )
