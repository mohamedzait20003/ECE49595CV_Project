from pathlib import Path
from typing import Sequence

import torch
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image

from src.data import Cars196Dataset



def _annotate_and_save(
    dataset: Cars196Dataset,
    indices: Sequence[int],
    split: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        image = image.convert("RGB")

        width, height = image.size
        label = dataset.class_names[target] if target >= 0 and target < len(dataset.class_names) else "unlabeled"
        safe_label = label.replace(" ", "_")

        draw = ImageDraw.Draw(image)
        text = f"{split} | {label} | {width}x{height}"
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), text)
            text_w, text_h = x1 - x0, y1 - y0
        except AttributeError:
            # Fallback for older Pillow versions.
            text_w, text_h = draw.textsize(text)
        # Add a small background box for readability.
        draw.rectangle([0, 0, text_w + 8, text_h + 8], fill=(0, 0, 0))
        draw.text((4, 4), text, fill=(255, 255, 255))

        fname = out_dir / f"{split}_{i:02d}_{safe_label}.jpg"
        image.save(fname)
        print(f"Saved {fname} ({text})")


def main() -> None:
    data_root = Path("archive")
    output_dir = Path("outputs") / "samples"

    train_ds = Cars196Dataset(
        root=data_root,
        split="train",
        transform=None,  # keep PIL for inspection
        use_default_transform=False,
        crop_mode="bbox",
        pad_to_square=True,
    )
    test_ds = Cars196Dataset(
        root=data_root,
        split="test",
        transform=None,
        use_default_transform=False,
        crop_mode="bbox",
        pad_to_square=True,
    )

    num_train = min(10, len(train_ds))
    num_test = min(10, len(test_ds))

    _annotate_and_save(train_ds, range(num_train), "train", output_dir)
    _annotate_and_save(test_ds, range(num_test), "test", output_dir)


if __name__ == "__main__":
    main()
