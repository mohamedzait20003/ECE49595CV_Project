"""
Quick integrity check for Cars-style datasets.
Checks that numeric files 00001.jpg ... N.jpg exist in train/test folders
and reports any missing indices.
"""
import argparse
from pathlib import Path
import json


def collect_indices(folder: Path):
    idxs = []
    for p in folder.glob("*.jpg"):
        name = p.stem  # e.g., 00001
        if name.isdigit():
            idxs.append(int(name))
    return sorted(idxs)


def report_split(split_name: str, folder: Path, expected: int):
    idxs = collect_indices(folder)
    missing = [i for i in range(1, expected + 1) if i not in idxs] if expected else []

    if not idxs:
        print(f"[{split_name}] No JPG files found in {folder}")
    else:
        print(f"[{split_name}] found {len(idxs)} files; expected {expected}")
        print(f"[{split_name}] min idx: {idxs[0]}, max idx: {idxs[-1]}")
        if missing:
            print(f"[{split_name}] missing {len(missing)} indices (first 50 shown): {missing[:50]}")
        else:
            print(f"[{split_name}] no missing indices detected.")

    return {
        "split": split_name,
        "folder": str(folder),
        "expected": expected,
        "found": len(idxs),
        "min_idx": idxs[0] if idxs else None,
        "max_idx": idxs[-1] if idxs else None,
        "missing": missing,
    }


def main():
    ap = argparse.ArgumentParser(description="Check missing indices in Cars train/test splits.")
    ap.add_argument("--data-root", type=Path, default=None, help="Optional common root. If omitted, uses defaults below.")
    ap.add_argument("--train-dir", dest="train_dir", type=str, default=None, help="Train subdir (optional).")
    ap.add_argument("--test-dir", dest="test_dir", type=str, default=None, help="Test subdir (optional).")
    ap.add_argument("--expected-train", type=int, default=None, help="Expected count for train split.")
    ap.add_argument("--expected-test", type=int, default=None, help="Expected count for test split.")
    ap.add_argument("--save-path", type=Path, default=Path("missing_indices.json"), help="Where to save JSON report.")
    ap.add_argument("--no-save", action="store_true", help="Do not write a report to disk.")
    args = ap.parse_args()

    # Defaults baked for local noise sets if no args are provided.
    default_train = Path(r"C:\Users\trevo\Documents\projects\CV term\clean\noise_train")
    default_test = Path(r"C:\Users\trevo\Documents\projects\CV term\clean\noise_test")
    default_train_expected = 8136
    default_test_expected = 8034

    if args.data_root or args.train_dir or args.test_dir or args.expected_train or args.expected_test:
        # Custom paths provided.
        data_root = args.data_root or Path(".")
        train_dir = Path(args.train_dir) if args.train_dir else Path("cars_train/cars_train")
        test_dir = Path(args.test_dir) if args.test_dir else Path("cars_test/cars_test")
        expected_train = args.expected_train if args.expected_train is not None else 8144
        expected_test = args.expected_test if args.expected_test is not None else 8041
        train_folder = data_root / train_dir
        test_folder = data_root / test_dir
    else:
        # Use baked defaults for noise train/test on Windows path.
        train_folder = default_train
        test_folder = default_test
        expected_train = default_train_expected
        expected_test = default_test_expected

    print(f"Checking train -> {train_folder}")
    train_result = report_split("train", train_folder, expected_train)
    print(f"\nChecking test  -> {test_folder}")
    test_result = report_split("test", test_folder, expected_test)

    if not args.no_save:
        save_path = args.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"train": train_result, "test": test_result}
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved report -> {save_path}")


if __name__ == "__main__":
    main()
