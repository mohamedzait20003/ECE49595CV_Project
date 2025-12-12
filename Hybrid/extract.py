import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from src.data import build_cars196_train_test_loaders
from src.models import CNNTransformerHybrid, HybridConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CNN features and class-token attention.")
    parser.add_argument("--data-root", type=str, default="archive")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/extract")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint (.pt) to load.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(args: argparse.Namespace, num_classes: int) -> CNNTransformerHybrid:
    cfg = HybridConfig(
        backbone=args.backbone,
        num_classes=num_classes,
        freeze_backbone=False,  # allow full forward regardless of requires_grad
    )
    model = CNNTransformerHybrid(cfg)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model.to(torch.device(args.device))


def upsample_map(feature_map: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(feature_map, size=target_hw, mode="bilinear", align_corners=False)


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def save_feature_vis(
    images: torch.Tensor,
    feats: torch.Tensor,
    attn: torch.Tensor,
    class_names: List[str],
    indices: List[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    b = images.size(0)
    images_vis = unnormalize(images)
    for i in range(b):
        label = class_names[indices[i]] if indices[i] < len(class_names) else f"class_{indices[i]}"
        base = out_dir / f"sample_{i:02d}_{label.replace(' ', '_')}"

        # Save input image
        save_image(images_vis[i].clamp(0, 1), base.with_suffix(".jpg"))

        # Save mean activation map
        mean_map = feats[i].mean(dim=0, keepdim=True)  # [1, H, W]
        mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)
        mean_up = upsample_map(mean_map.unsqueeze(0), target_hw=(images.size(2), images.size(3)))[0]
        save_image(mean_up, base.with_name(base.stem + "_feat_mean.jpg"))

        # Save a few channels
        for ch in range(min(4, feats.size(1))):
            ch_map = feats[i, ch : ch + 1]
            ch_map = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min() + 1e-6)
            ch_up = upsample_map(ch_map.unsqueeze(0), target_hw=(images.size(2), images.size(3)))[0]
            save_image(ch_up, base.with_name(base.stem + f"_feat_ch{ch}.jpg"))

        # Save class-token attention if present
        if attn is not None:
            cls_attn = attn[i]
            if cls_attn.dim() == 4:
                # [heads, tgt, src] or [heads, tgt, src] depending on layout; assume [heads, tgt, src]
                cls_attn = cls_attn[:, 0, 1:].mean(dim=0)  # [N]
            elif cls_attn.dim() == 3:
                # [tgt, src] aggregated
                cls_attn = cls_attn[0, 1:]
            else:
                # Unexpected shape; skip
                cls_attn = None

            if cls_attn is not None:
                num_tokens = cls_attn.numel()
                hw = int(num_tokens ** 0.5)
                cls_attn = cls_attn.reshape(1, 1, hw, hw)
                cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-6)
                cls_up = upsample_map(cls_attn, target_hw=(images.size(2), images.size(3)))[0]
                save_image(cls_up, base.with_name(base.stem + "_attn_cls.jpg"))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.output_dir)

    train_loader, test_loader, class_names = build_cars196_train_test_loaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=2,
        crop_mode="bbox",
        pad_to_square=True,
    )

    model = load_model(args, num_classes=len(class_names))
    device = torch.device(args.device)

    # Use test loader for visualization.
    dataset = test_loader.dataset
    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    images = []
    targets = []
    for idx in indices:
        img, tgt = dataset[idx]
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    with torch.no_grad():
        logits, feats, attn = model(images, return_features=True, return_attn=True)

    save_feature_vis(
        images.cpu(),
        feats.cpu(),
        attn.cpu() if attn is not None else None,
        class_names,
        targets_tensor.tolist(),
        out_dir,
    )
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
