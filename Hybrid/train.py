import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import build_cars196_train_test_loaders
from src.models import CNNTransformerHybrid, HybridConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+Transformer hybrid on Cars196.")
    parser.add_argument("--data-root", type=str, default="archive", help="Path to Cars196 data root.")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-freeze-backbone", action="store_true", help="Unfreeze backbone for finetuning.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-ckpt", type=str, default=None, help="Path to save final model state_dict (.pt).")
    return parser.parse_args()


def build_model(cfg: HybridConfig, device: torch.device) -> CNNTransformerHybrid:
    model = CNNTransformerHybrid(cfg)
    model.to(device)
    return model


def run_epoch(
    model: CNNTransformerHybrid,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        if step % 20 == 0:
            print(f"Epoch {epoch} step {step}: loss={loss.item():.4f}")
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: CNNTransformerHybrid, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
    return running_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_loader, test_loader, class_names = build_cars196_train_test_loaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_mode="bbox",
        pad_to_square=True,
    )

    cfg = HybridConfig(
        backbone=args.backbone,
        num_classes=len(class_names),
        freeze_backbone=not args.no_freeze_backbone,
    )
    model = build_model(cfg, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch} done: train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%")

    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        if ckpt_path.suffix == "":
            ckpt_path = ckpt_path.with_suffix(".pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
