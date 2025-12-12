from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class CarAnnotation:
    fname: str
    bbox: Tuple[int, int, int, int]
    label: Optional[int]  # zero-based, None for test set


def default_transforms(is_train: bool) -> transforms.Compose:
    """ImageNet-style preprocessing with light augmentation on train."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _resolve_split_dir(root: Path, split_dir: str) -> Path:
    """Handle the nested cars_train/cars_train pattern in the archive."""
    base = root / split_dir
    nested = base / split_dir
    if nested.is_dir():
        return nested
    if base.is_dir():
        return base
    raise FileNotFoundError(f"Could not find split directory {split_dir} under {root}")


def load_class_names(root: Path) -> List[str]:
    meta_path = root / "car_devkit" / "devkit" / "cars_meta.mat"
    mat = loadmat(meta_path, squeeze_me=True, struct_as_record=False)
    names = mat.get("class_names")
    if isinstance(names, np.ndarray):
        names = names.squeeze().tolist()
    if not isinstance(names, (list, tuple)):
        names = [names]
    return [str(n[0] if isinstance(n, np.ndarray) else n) for n in names]


def _as_list(annos: Iterable) -> List:
    if isinstance(annos, np.ndarray):
        annos = annos.reshape(-1).tolist()
    if not isinstance(annos, (list, tuple)):
        annos = [annos]
    return [a for a in annos if a is not None]


def load_annotations(root: Path, split: str) -> List[CarAnnotation]:
    split_key = "train" if split != "test" else "test"
    mat_path = root / "car_devkit" / "devkit" / f"cars_{split_key}_annos.mat"
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    annos_raw = mat["annotations"]
    annos = _as_list(annos_raw)

    parsed: List[CarAnnotation] = []
    for anno in annos:
        fname = str(anno.fname)
        bbox = (
            int(anno.bbox_x1),
            int(anno.bbox_y1),
            int(anno.bbox_x2),
            int(anno.bbox_y2),
        )
        label_field = getattr(anno, "class", getattr(anno, "class_", None))
        label = int(label_field) - 1 if label_field is not None else None
        parsed.append(CarAnnotation(fname=fname, bbox=bbox, label=label))

    return parsed


class Cars196Dataset(Dataset):
    """Cars196 dataset with optional bbox cropping (stand-in for SAM crops)."""

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        indices: Optional[Sequence[int]] = None,
        transform: Optional[transforms.Compose] = None,
        use_default_transform: bool = True,
        crop_mode: str = "bbox",
        crops_root: Optional[Path | str] = None,
        annotations: Optional[List[CarAnnotation]] = None,
        class_names: Optional[List[str]] = None,
        pad_to_square: bool = True,
        pad_fill: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.crop_mode = crop_mode
        self.crops_root = Path(crops_root) if crops_root is not None else None
        if transform is not None:
            self.transform = transform
        elif use_default_transform:
            self.transform = default_transforms(is_train=(split == "train"))
        else:
            self.transform = None
        self.pad_to_square = pad_to_square
        self.pad_fill = pad_fill

        if crop_mode not in {"bbox", "none", "sam"}:
            raise ValueError(f"crop_mode must be 'bbox', 'none', or 'sam', got {crop_mode}")

        self.class_names = class_names or load_class_names(self.root)
        self.annotations = annotations or load_annotations(self.root, split)

        self.samples = self.annotations if indices is None else [self.annotations[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = self._load_image(sample)
        target = sample.label if sample.label is not None else -1

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def _resolve_image_path(self, fname: str) -> Path:
        split_dir = "cars_train" if self.split != "test" else "cars_test"
        if self.crop_mode == "sam" and self.crops_root is not None:
            sam_dir = _resolve_split_dir(self.crops_root, split_dir)
            candidate = sam_dir / fname
            if candidate.exists():
                return candidate
        base_dir = _resolve_split_dir(self.root, split_dir)
        return base_dir / fname

    @staticmethod
    def _crop_to_bbox(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        width, height = image.size
        x1 = max(int(bbox[0]) - 1, 0)
        y1 = max(int(bbox[1]) - 1, 0)
        x2 = min(int(bbox[2]), width)
        y2 = min(int(bbox[3]), height)
        if x2 <= x1 or y2 <= y1:
            return image
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _pad_to_square(image: Image.Image, fill: Tuple[int, int, int]) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        if width > height:
            delta = width - height
            padding = (0, delta // 2, 0, delta - delta // 2)
        else:
            delta = height - width
            padding = (delta // 2, 0, delta - delta // 2, 0)
        return ImageOps.expand(image, border=padding, fill=fill)

    def _load_image(self, sample: CarAnnotation) -> Image.Image:
        image_path = self._resolve_image_path(sample.fname)
        image = Image.open(image_path).convert("RGB")
        if self.crop_mode == "bbox":
            image = self._crop_to_bbox(image, sample.bbox)
        if self.pad_to_square:
            image = self._pad_to_square(image, self.pad_fill)
        return image


def build_cars196_loaders(
    data_root: Path | str,
    batch_size: int = 64,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    crop_mode: str = "bbox",
    crops_root: Optional[Path | str] = None,
    pad_to_square: bool = True,
    pad_fill: Tuple[int, int, int] = (0, 0, 0),
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], List[str]]:
    """
    Create train/val dataloaders. Val set is a random split from train annotations.

    data_root should point to the folder containing cars_train, cars_test, and car_devkit.
    """
    root = Path(data_root)
    class_names = load_class_names(root)
    train_annos = load_annotations(root, "train")

    indices = list(range(len(train_annos)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Cars196Dataset(
        root=root,
        split="train",
        indices=train_indices,
        annotations=train_annos,
        class_names=class_names,
        transform=default_transforms(is_train=True),
        crop_mode=crop_mode,
        crops_root=crops_root,
        pad_to_square=pad_to_square,
        pad_fill=pad_fill,
    )

    val_loader: Optional[DataLoader] = None
    if val_size > 0:
        val_dataset = Cars196Dataset(
            root=root,
            split="train",
            indices=val_indices,
            annotations=train_annos,
            class_names=class_names,
            transform=default_transforms(is_train=False),
            crop_mode=crop_mode,
            crops_root=crops_root,
            pad_to_square=pad_to_square,
            pad_fill=pad_fill,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )

    return train_loader, val_loader, class_names


def build_cars196_train_test_loaders(
    data_root: Path | str,
    batch_size: int = 64,
    num_workers: int = 4,
    crop_mode: str = "bbox",
    crops_root: Optional[Path | str] = None,
    pad_to_square: bool = True,
    pad_fill: Tuple[int, int, int] = (0, 0, 0),
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train and test loaders; test split is used for evaluation (no extra val split).
    """
    root = Path(data_root)
    class_names = load_class_names(root)

    train_dataset = Cars196Dataset(
        root=root,
        split="train",
        transform=default_transforms(is_train=True),
        crop_mode=crop_mode,
        crops_root=crops_root,
        pad_to_square=pad_to_square,
        pad_fill=pad_fill,
    )
    test_dataset = Cars196Dataset(
        root=root,
        split="test",
        transform=default_transforms(is_train=False),
        crop_mode=crop_mode,
        crops_root=crops_root,
        pad_to_square=pad_to_square,
        pad_fill=pad_fill,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
    )

    return train_loader, test_loader, class_names
