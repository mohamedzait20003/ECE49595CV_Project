import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision.utils import save_image, make_grid
from torchvision import models

from src.data import build_cars196_train_test_loaders, default_transforms
from src.models import CNNTransformerHybrid, HybridConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feature maps across ResNet backbones.")
    parser.add_argument("--data-root", type=str, default="archive")
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        default=["resnet18", "resnet34", "resnet50"],
        help=(
            "Backbones to compare, e.g., resnet18 resnet34 resnet50 resnet101 resnet152 "
            "squeezenet1_0 squeezenet1_1 mobilenet_v2 mobilenet_v3_small mobilenet_v3_large "
            "shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0 "
            "regnet_x_200mf regnet_x_400mf efficientnet_b0 convnext_tiny"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=1, help="Number of images to visualize.")
    parser.add_argument("--top-k", type=int, default=7, help="Top-k channels by global mean activation.")
    parser.add_argument("--score", type=str, default="mean", choices=["mean", "max"], help="Channel scoring method.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="outputs/compare.jpg")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional hybrid checkpoint to load.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tap-stages",
        type=str,
        nargs="+",
        default=[],
        help="If set (e.g., layer1 layer2 layer3 layer4), compare stages from a single ResNet backbone.",
    )
    return parser.parse_args()


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def load_resnet_hybrid(backbone: str, num_classes: int, device: torch.device, checkpoint: str | None) -> CNNTransformerHybrid:
    cfg = HybridConfig(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=False,
    )
    model = CNNTransformerHybrid(cfg).to(device).eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    return model


def load_cnn_feature_extractor(backbone: str, device: torch.device):
    """
    Load a backbone that outputs a feature map (no transformer head).
    Returns (model, forward_fn) where forward_fn returns feature map.
    """
    backbone = backbone.lower()

    def eval_and_freeze(m):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m.to(device)

    if backbone == "squeezenet1_0":
        m = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            return m.features(x)

    elif backbone == "squeezenet1_1":
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            return m.features(x)

    elif backbone == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        m = eval_and_freeze(m)

        def forward(x):
            return m.features(x)

    elif backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            return m.features(x)

    elif backbone == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m = eval_and_freeze(m)

        def forward(x):
            return m.features(x)

    elif backbone == "shufflenet_v2_x0_5":
        m = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            x = m.conv1(x)
            x = m.maxpool(x)
            x = m.stage2(x)
            x = m.stage3(x)
            x = m.stage4(x)
            x = m.conv5(x)
            return x

    elif backbone == "shufflenet_v2_x1_0":
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            x = m.conv1(x)
            x = m.maxpool(x)
            x = m.stage2(x)
            x = m.stage3(x)
            x = m.stage4(x)
            x = m.conv5(x)
            return x

    elif backbone == "shufflenet_v2_x1_5":
        m = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            x = m.conv1(x)
            x = m.maxpool(x)
            x = m.stage2(x)
            x = m.stage3(x)
            x = m.stage4(x)
            x = m.conv5(x)
            return x

    elif backbone == "shufflenet_v2_x2_0":
        m = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            x = m.conv1(x)
            x = m.maxpool(x)
            x = m.stage2(x)
            x = m.stage3(x)
            x = m.stage4(x)
            x = m.conv5(x)
            return x

    elif backbone == "regnet_x_200mf":
        m = models.regnet_x_200mf(weights=models.RegNet_X_200MF_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            x = m.stem(x)
            x = m.trunk_output(x)
            return x

    elif backbone == "regnet_x_400mf":
        m = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            x = m.stem(x)
            x = m.trunk_output(x)
            return x

    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            return m.features(x)

    elif backbone == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m = eval_and_freeze(m)

        def forward(x):
            return m.features(x)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return m, forward


def select_topk_channels(feats: torch.Tensor, k: int, method: str = "mean") -> List[int]:
    # feats: [C, H, W]
    if method == "max":
        scores = feats.amax(dim=[1, 2])
    else:
        scores = feats.mean(dim=[1, 2])  # global mean
    topk = torch.topk(scores, min(k, feats.size(0))).indices.tolist()
    return topk


def prepare_views(
    images: torch.Tensor,
    feats: torch.Tensor,
    attn: torch.Tensor | None,
    topk: int,
    score_method: str,
    img_hw: Tuple[int, int],
) -> List[torch.Tensor]:
    views = []
    # Input
    views.append(images.clamp(0, 1))
    # Mean activation
    mean_map = feats.mean(dim=0, keepdim=True)
    mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)
    mean_up = torch.nn.functional.interpolate(mean_map.unsqueeze(0), size=img_hw, mode="bilinear", align_corners=False)[0]
    views.append(mean_up)
    # Top-k channels
    channel_indices = select_topk_channels(feats, topk, method=score_method)
    for ch in channel_indices:
        ch_map = feats[ch : ch + 1]
        ch_map = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min() + 1e-6)
        ch_up = torch.nn.functional.interpolate(ch_map.unsqueeze(0), size=img_hw, mode="bilinear", align_corners=False)[0]
        views.append(ch_up)
    # Attention
    if attn is not None:
        cls_attn = attn
        if cls_attn.dim() == 4:
            cls_attn = cls_attn[:, 0, 1:].mean(dim=0)
        elif cls_attn.dim() == 3:
            cls_attn = cls_attn[0, 1:]
        else:
            cls_attn = None
        if cls_attn is not None:
            num_tokens = cls_attn.numel()
            hw = int(num_tokens ** 0.5)
            cls_attn = cls_attn.reshape(1, 1, hw, hw)
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-6)
            cls_up = torch.nn.functional.interpolate(cls_attn, size=img_hw, mode="bilinear", align_corners=False)[0]
            views.append(cls_up)
    # Ensure 3-channel for grid stacking.
    views_rgb = []
    for v in views:
        if v.shape[0] == 1:
            v = v.repeat(3, 1, 1)
        views_rgb.append(v)
    return views_rgb


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)
    out_path = Path(args.output)

    train_loader, test_loader, class_names = build_cars196_train_test_loaders(
        data_root=Path(args.data_root),
        batch_size=1,
        num_workers=2,
        crop_mode="bbox",
        pad_to_square=True,
    )
    dataset = test_loader.dataset
    idx = random.randint(0, len(dataset) - 1)
    image, target = dataset[idx]
    image = image.unsqueeze(0).to(device)
    img_hw = (image.shape[2], image.shape[3])
    image_vis = unnormalize(image)

    rows: List[torch.Tensor] = []
    resnet_names = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}

    if args.tap_stages:
        # Compare stages of the first backbone (expect a ResNet).
        backbone = args.backbones[0].lower()
        if backbone not in resnet_names:
            raise ValueError("tap-stages mode requires a ResNet backbone (e.g., resnet50).")
        model = load_resnet_hybrid(backbone, num_classes=len(class_names), device=device, checkpoint=args.checkpoint)
        for stage in args.tap_stages:
            attn_map = None
            with torch.no_grad():
                logits, feats, attn = model(image, return_features=True, return_attn=True, tap_stage=stage)
            feats_cpu = feats.cpu() if isinstance(feats, torch.Tensor) else feats[0].cpu()
            if feats_cpu.dim() == 4:
                feats_cpu = feats_cpu[0]
            if stage == "layer4" and attn is not None:
                attn_map = attn[0].cpu()
            views = prepare_views(image_vis[0].cpu(), feats_cpu, attn_map, args.top_k, args.score, img_hw)
            row = torch.stack(views, dim=0)
            rows.append(row)
    else:
        for backbone in args.backbones:
            backbone_lower = backbone.lower()
            attn_map = None
            if backbone_lower in resnet_names:
                model = load_resnet_hybrid(backbone_lower, num_classes=len(class_names), device=device, checkpoint=args.checkpoint)
                with torch.no_grad():
                    logits, feats, attn = model(image, return_features=True, return_attn=True)
                feats = feats[0].cpu()
                attn_map = attn[0].cpu() if attn is not None else None
            else:
                model, forward_fn = load_cnn_feature_extractor(backbone_lower, device=device)
                with torch.no_grad():
                    feats = forward_fn(image).cpu()

            # Ensure feats is [C, H, W]
            if feats.dim() == 4:
                feats = feats[0]

            views = prepare_views(image_vis[0].cpu(), feats, attn_map, args.top_k, args.score, img_hw)
            row = torch.stack(views, dim=0)
            rows.append(row)

    # Pad rows to same number of columns (in case attention missing)
    max_cols = max(r.shape[0] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[0] < max_cols:
            pad_count = max_cols - r.shape[0]
            pad = torch.zeros((pad_count, *r.shape[1:]))
            r = torch.cat([r, pad], dim=0)
        padded_rows.append(r)

    # Stack rows vertically: each row is concatenated horizontally.
    row_images = [make_grid(r, nrow=r.shape[0], padding=2) for r in padded_rows]
    grid = torch.cat(row_images, dim=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    print(f"Saved compare grid to {out_path}")


if __name__ == "__main__":
    main()
