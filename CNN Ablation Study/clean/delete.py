"""
Delete JPG files by index (e.g., 00001.jpg) across one or more datasets.
Reads a JSON report with missing/bad indices (e.g., from count.py) and removes
matching files from specified train/test folders.
"""
import argparse
import json
from pathlib import Path


def delete_indices(folder: Path, indices):
    removed = []
    for idx in indices:
        fname = f"{idx:05d}.jpg"
        fpath = folder / fname
        if fpath.exists():
            fpath.unlink()
            removed.append(fname)
    return removed


def main():
    ap = argparse.ArgumentParser(description="Delete bad indices across datasets.")
    ap.add_argument("--report", type=Path, required=True, help="Path to JSON from count.py (with 'train'/'test' missing lists).")
    ap.add_argument("--train-dir", type=Path, required=True, help="Train folder to delete from (contains numbered JPGs).")
    ap.add_argument("--test-dir", type=Path, required=True, help="Test folder to delete from (contains numbered JPGs).")
    ap.add_argument("--no-train", action="store_true", help="Skip deleting from train.")
    ap.add_argument("--no-test", action="store_true", help="Skip deleting from test.")
    args = ap.parse_args()

    data = json.loads(args.report.read_text())
    bad_train = data.get("train", {}).get("missing", []) or []
    bad_test = data.get("test", {}).get("missing", []) or []

    if not args.no_train:
        removed_train = delete_indices(args.train_dir, bad_train)
        print(f"Removed {len(removed_train)} files from train ({args.train_dir})")
    if not args.no_test:
        removed_test = delete_indices(args.test_dir, bad_test)
        print(f"Removed {len(removed_test)} files from test ({args.test_dir})")


if __name__ == "__main__":
    main()
