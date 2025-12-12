from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BackboneName = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


@dataclass
class HybridConfig:
    backbone: BackboneName = "resnet34"
    embed_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    num_classes: int = 196
    freeze_backbone: bool = True
    pretrained: bool = True
    pos_patch: Optional[Tuple[int, int]] = None  # (H, W) tokens; if None, infer 7x7


def _build_backbone(name: BackboneName, pretrained: bool) -> Tuple[nn.Module, int, Tuple[int, int]]:
    """Return backbone (without avgpool/fc), output channels, and expected spatial size."""
    if name == "resnet18":
        builder = models.resnet18
    elif name == "resnet34":
        builder = models.resnet34
    elif name == "resnet50":
        builder = models.resnet50
    elif name == "resnet101":
        builder = models.resnet101
    elif name == "resnet152":
        builder = models.resnet152
    else:
        raise ValueError(f"Unsupported backbone {name}")

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    elif name == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V1
    else:
        weights = models.ResNet152_Weights.IMAGENET1K_V1
    weights = weights if pretrained else None
    backbone = builder(weights=weights)
    # Drop avgpool and fc; keep conv1..layer4.
    backbone.fc = nn.Identity()
    backbone.avgpool = nn.Identity()
    # Channels and nominal feature map size for 224x224 input.
    out_channels = 512 if name in {"resnet18", "resnet34"} else 2048
    feat_hw = (7, 7)
    return backbone, out_channels, feat_hw


class CNNTransformerHybrid(nn.Module):
    """
    CNN backbone -> projection -> cls token + pos embed -> Transformer encoder -> classifier.

    This keeps the backbone frozen by default for depth ablation; call `unfreeze_backbone`
    to fine-tune.
    """

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.config = config

        backbone, out_channels, feat_hw = _build_backbone(config.backbone, config.pretrained)
        self.backbone = backbone
        self.backbone_out_channels = out_channels
        self.backbone_feat_hw = config.pos_patch or feat_hw

        embed_dim = config.embed_dim
        self.proj = nn.Linear(out_channels, embed_dim)

        num_patches = self.backbone_feat_hw[0] * self.backbone_feat_hw[1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_ratio * embed_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, config.num_classes)

        self._init_parameters()
        if config.freeze_backbone:
            self.freeze_backbone()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    @torch.no_grad()
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def unfreeze_backbone(self, trainable_stages: Optional[Tuple[str, ...]] = None) -> None:
        """
        Unfreeze entire backbone or selected stages (e.g., ('layer4',)).
        """
        if trainable_stages is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        for name, module in self.backbone.named_children():
            if name in trainable_stages:
                for p in module.parameters():
                    p.requires_grad = True
            else:
                for p in module.parameters():
                    p.requires_grad = False

    def _forward_backbone(self, x: torch.Tensor, tap_stage: str = "layer4") -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Manual forward to stop before avgpool/fc.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if tap_stage == "layer1":
            h, w = x.shape[2], x.shape[3]
            return x, (h, w)

        x = self.backbone.layer2(x)
        if tap_stage == "layer2":
            h, w = x.shape[2], x.shape[3]
            return x, (h, w)

        x = self.backbone.layer3(x)
        if tap_stage == "layer3":
            h, w = x.shape[2], x.shape[3]
            return x, (h, w)

        x = self.backbone.layer4(x)
        h, w = x.shape[2], x.shape[3]
        return x, (h, w)

    def _interp_pos_embed(self, pos_embed: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """
        Interpolate positional embeddings if feature map spatial dims differ from expected.
        """
        B, L, C = pos_embed.shape  # B should be 1
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        src_h, src_w = self.backbone_feat_hw
        patch_pos = patch_pos.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=hw, mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, hw[0] * hw[1], C)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attn: bool = False,
        tap_stage: str = "layer4",
    ):
        feats, hw = self._forward_backbone(x, tap_stage=tap_stage)
        b, c, h, w = feats.shape
        # If tapping before layer4, skip transformer/head.
        if tap_stage != "layer4":
            if return_attn or return_features:
                return None, feats, None
            raise ValueError("tap_stage != 'layer4' not supported for classification logits")

        feats_tokens = feats.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.proj(feats_tokens)  # [B, N, D]

        cls_tokens = self.cls_token.expand(b, -1, -1)
        pos_embed = self.pos_embed
        if hw != self.backbone_feat_hw:
            pos_embed = self._interp_pos_embed(pos_embed, hw)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + pos_embed
        tokens = self.pos_drop(tokens)

        attn_weights = None
        if return_attn:
            # Hook into attention to capture class-token attention from last layer.
            attn_maps = []

            def _hook(mod, inp, out):
                # out may be tensor or (attn_output, attn_weights)
                attn = out
                if isinstance(attn, tuple) and len(attn) > 1:
                    attn = attn[1]
                if attn is not None:
                    attn_maps.append(attn.detach())

            handle = self.transformer.layers[-1].self_attn.register_forward_hook(_hook)
            tokens = self.transformer(tokens)
            handle.remove()
            if attn_maps:
                attn_weights = attn_maps[0]
        else:
            tokens = self.transformer(tokens)

        cls_out = self.norm(tokens[:, 0])
        logits = self.head(cls_out)

        if return_attn or return_features:
            return logits, feats, attn_weights
        return logits
