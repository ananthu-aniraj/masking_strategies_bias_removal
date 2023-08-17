import torch
import torch.nn as nn


class LinearHeadViTTimm(nn.Module):
    def __init__(self, *, backbone:nn.Module, linear_head: nn.Module, patch_tokens_only: bool = False, class_token_only: bool = False):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.patch_tokens_only = patch_tokens_only
        self.class_token_only = class_token_only

    def forward(self, x):
        x = self.backbone.forward_features(x)
        cls_token = x[:, 0, :]
        patch_tokens = x[:, 1:, :]
        patch_tokens = patch_tokens.mean(dim=1)
        if self.patch_tokens_only:
            linear_input = patch_tokens
        elif self.class_token_only:
            linear_input = cls_token
        else:
            linear_input = torch.cat([cls_token, patch_tokens], dim=1)
        out = self.linear_head(linear_input)
        return out


def linear_head_vit_dino_timm(backbone: nn.Module, embed_dim: int, num_classes: int, patch_tokens_only: bool = False, class_token_only: bool = False):
    if patch_tokens_only or class_token_only:
        linear_head = nn.Linear(embed_dim, num_classes)
    else:
        linear_head = nn.Linear(2 * embed_dim, num_classes)
    return LinearHeadViTTimm(backbone=backbone, linear_head=linear_head, patch_tokens_only=patch_tokens_only, class_token_only=class_token_only)
