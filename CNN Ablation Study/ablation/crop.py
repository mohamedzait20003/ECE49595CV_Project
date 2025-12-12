#FIX
# Fine-tune ResNet-50 on a bbox-cropped Cars variant with baseline-style RRC.
# Method:
# - Pad bbox to a square (with tolerance), keep native padded resolution (no downsample).
# - Train: treat the padded square as the full image and apply RandomResizedCrop
#   to 224 with scale=(0.1, 1.0) and ratio=(3/4, 4/3), flip + normalize.
# - Val: resize padded square to 256, center crop 224, normalize.
import argparse
import json
import math
from typing import Optional
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm


def load_class_names(meta_path: Path):
    meta = scipy.io.loadmat(meta_path)
    return [c[0][0] for c in meta["class_names"][0]]


def load_train_annos(anno_path: Path):
    mat = scipy.io.loadmat(anno_path)
    annos = []
    for entry in mat["annotations"][0]:
        annos.append(
            {
                "fname": entry["fname"][0],
                "class": int(entry["class"][0][0]) - 1,
                "bbox": [
                    float(entry["bbox_x1"][0][0]),
                    float(entry["bbox_y1"][0][0]),
                    float(entry["bbox_x2"][0][0]),
                    float(entry["bbox_y2"][0][0]),
                ],
            }
        )
    return annos


def load_test_annos(anno_path: Path):
    return load_train_annos(anno_path)


def pad_square_crop(img: Image.Image, bbox, tolerance: float, resize_size: Optional[int]):
    """
    Pad bbox to square (with tolerance), crop from image with black padding if needed,
    and (optionally) resize to resize_size. Returns (square_img, cx, cy, bw, bh) in resized coords.
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0

    size = max(bw, bh) * (1 + 2 * tolerance)
    if size < 1:
        size = 1.0
    half = size / 2.0
    left = cx - half
    top = cy - half
    right = cx + half
    bottom = cy + half

    dst_size = int(round(size))
    canvas = Image.new("RGB", (dst_size, dst_size), (0, 0, 0))

    src_left = max(0, int(math.floor(left)))
    src_top = max(0, int(math.floor(top)))
    src_right = min(w, int(math.ceil(right)))
    src_bottom = min(h, int(math.ceil(bottom)))

    if src_right > src_left and src_bottom > src_top:
        src_crop = img.crop((src_left, src_top, src_right, src_bottom))
        paste_x = src_left - int(round(left))
        paste_y = src_top - int(round(top))
        canvas.paste(src_crop, (paste_x, paste_y))

    # If resize_size is None, keep the native padded resolution (no downsample before aug)
    if resize_size is None:
        resize_size = dst_size
        scale = 1.0
    else:
        scale = resize_size / float(dst_size)
        canvas = canvas.resize((resize_size, resize_size), Image.BILINEAR)
    cx_r = (cx - left) * scale
    cy_r = (cy - top) * scale
    bw_r = bw * scale
    bh_r = bh * scale
    return canvas, cx_r, cy_r, bw_r, bh_r


class CarsDataset(Dataset):
    def __init__(
        self,
        samples,
        transform=None,
        tolerance=0.1,
        base_size=None,
    ):
        self.samples = samples
        self.transform = transform
        self.tolerance = tolerance
        self.base_size = base_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, bbox = self.samples[idx]
        img = Image.open(path).convert("RGB")
        square, _, _, _, _ = pad_square_crop(
            img, bbox, self.tolerance, self.base_size
        )
        if self.transform:
            square = self.transform(square)
        return square, label


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        return (pred == target).sum().item() / target.size(0)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            bsz = targets.size(0)
            loss_sum += loss.item() * bsz
            acc_sum += accuracy(outputs, targets) * bsz
            n += bsz
    if n == 0:
        return float("nan"), 0.0
    return loss_sum / n, acc_sum / n


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)
    train_dir = data_root / "cars_train" / "cars_train"
    devkit = data_root / "car_devkit" / "devkit"
    test_dir = data_root / "cars_test" / "cars_test"

    class_names = load_class_names(devkit / "cars_meta.mat")
    num_classes = len(class_names)
    annos = load_train_annos(devkit / "cars_train_annos.mat")

    samples = [
        (train_dir / a["fname"], a["class"], tuple(a["bbox"]))
        for a in annos
        if (train_dir / a["fname"]).exists()
    ]

    if args.use_test_as_val:
        train_samples = samples
        test_annos = load_test_annos(devkit / "cars_test_annos.mat")
        val_samples = [
            (test_dir / a["fname"], a["class"], tuple(a["bbox"]))
            for a in test_annos
            if (test_dir / a["fname"]).exists()
        ]
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(samples)),
            test_size=args.val_split,
            stratify=[s[1] for s in samples],
            random_state=42,
        )
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.img_size, scale=(0.1, 1.0), ratio=(0.75, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    print(f"Classes: {num_classes}, Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    if args.use_test_as_val and len(val_samples) == 0:
        print("Warning: use_test_as_val was set but no test samples were found. Check devkit/test paths.")

    train_ds = CarsDataset(
        train_samples,
        transform=train_tf,
        tolerance=args.tolerance,
        base_size=None,
    )
    val_ds = CarsDataset(
        val_samples,
        transform=val_tf,
        tolerance=args.tolerance,
        base_size=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    best_acc = 0.0
    history = []
    run_dir = Path(args.ckpt_dir) / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(images)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            bsz = targets.size(0)
            running_loss += loss.item() * bsz
            running_acc += accuracy(outputs, targets) * bsz
            n += bsz
            pbar.set_postfix(
                loss=running_loss / n,
                acc=running_acc / n,
                lr=optimizer.param_groups[0]["lr"],
            )
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / n,
                "train_acc": running_acc / n,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = run_dir / "resnet50_best.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"Saved best -> {ckpt_path} (acc={val_acc:.4f})")

    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Best val acc:", best_acc)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune ResNet-50 on a bbox-cropped Cars variant (bbox-centered random zooms)."
    )
    p.add_argument("--data-root", type=Path, default=Path("/home/jut/archive"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--milestones", nargs="+", type=int, default=[30, 60])
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints_crop"))
    p.add_argument("--run-name", type=str, default=None, help="Optional subfolder under ckpt-dir for this run (defaults to timestamp).")
    p.add_argument(
        "--use-test-as-val",
        action="store_true",
        default=True,
        help="Use full train set for training and devkit test labels for validation (no train/val split).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Fractional padding around bbox (e.g., 0.1 adds 10%% each side).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
