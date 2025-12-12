import re
import torch
import kagglehub
from PIL import Image
import scipy.io as sio
from pathlib import Path
from torchvision import transforms
from typing import Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader


class Cars196Dataset(Dataset):
    """
    Stanford Cars Dataset (Cars196) with multi-label support for brand,
    model, and year.

    The dataset contains 196 classes of cars with annotations for
    training and testing.
    Each class name follows the format: "Make Model Year"
    (e.g., "AM General Hummer SUV 2000")

    This dataset class extracts three separate labels:
    - Brand (Make): e.g., "AM General", "Acura", "BMW"
    - Model: e.g., "Hummer SUV", "Integra Type R", "M3"
    - Year: e.g., 2000, 2001, 2012
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        auto_download_kaggle: bool = True,
        cache_dir: Optional[str] = None,
        label_mappings: Optional[Dict] = None
    ):
        """
        Args:
            root_dir: Root directory of the dataset. If None and
                     auto_download_kaggle=True, will download from Kaggle
                     automatically.
            split: 'train' or 'test'
            transform: Optional transform to be applied on images
            download: Whether to download the dataset if not found
            auto_download_kaggle: Automatically download from Kaggle
                                 using kagglehub
            cache_dir: Directory to cache downloaded dataset. If None,
                      uses default kagglehub cache location.
            label_mappings: Pre-built label mappings to share between
                           train/test splits. If None, builds new mappings.
        """
        self.split = split
        self.transform = transform

        # Download dataset if needed
        if root_dir is None and auto_download_kaggle:
            print("Downloading Stanford Cars Dataset from Kaggle...")
            if cache_dir:
                # Set cache directory for kagglehub
                import os
                os.environ['KAGGLEHUB_CACHE'] = cache_dir
            root_dir = kagglehub.dataset_download(
                "eduardo4jesus/stanford-cars-dataset"
            )
            print(f"Dataset downloaded to: {root_dir}")

        if root_dir is None:
            raise ValueError(
                "root_dir must be provided or "
                "auto_download_kaggle must be True"
            )

        self.root_dir = Path(root_dir)

        # Load annotations and setup paths
        self._load_annotations()

        # Build or use provided label mappings
        if label_mappings is not None:
            # Use shared label mappings
            self._use_label_mappings(label_mappings)
        else:
            # Build new label mappings
            self._build_label_mappings()

    def _load_annotations(self):
        """Load and parse annotations from the dataset."""
        # Look for annotation files
        annotation_file = None
        image_dir = None

        # Common paths in the Kaggle dataset
        possible_paths = [
            (self.root_dir / 'cars_annos.mat',
             self.root_dir / 'car_ims'),
            (self.root_dir / 'devkit' / 'cars_meta.mat',
             self.root_dir / 'cars_train'),
            (self.root_dir / f'cars_{self.split}_annos.mat',
             self.root_dir / f'cars_{self.split}'),
            (self.root_dir / 'devkit' / f'cars_{self.split}_annos.mat',
             self.root_dir / f'cars_{self.split}'),
        ]

        # Find the correct paths
        for anno_path, img_path in possible_paths:
            if anno_path.exists():
                annotation_file = anno_path
                if img_path.exists():
                    image_dir = img_path
                break

        # If standard structure not found, search for files
        if annotation_file is None:
            print(f"Searching for annotation files in {self.root_dir}...")
            anno_files = list(self.root_dir.rglob('*.mat'))
            if anno_files:
                files_str = [str(f) for f in anno_files]
                print(f"Found annotation files: {files_str}")
                # Prefer train/test specific annotations
                for af in anno_files:
                    if self.split in af.name:
                        annotation_file = af
                        break
                if annotation_file is None:
                    annotation_file = anno_files[0]

        if image_dir is None:
            # Search for image directories
            img_dirs = [
                d for d in self.root_dir.rglob('*')
                if d.is_dir() and 'car' in d.name.lower()
            ]
            if img_dirs:
                print(f"Found image directories: {[str(d) for d in img_dirs]}")
                for img_d in img_dirs:
                    if self.split in img_d.name or 'car_ims' in img_d.name:
                        image_dir = img_d
                        break
                if image_dir is None:
                    image_dir = img_dirs[0]

        if annotation_file is None or not annotation_file.exists():
            raise FileNotFoundError(
                f"Could not find annotation file for split "
                f"'{self.split}' in {self.root_dir}"
            )

        if image_dir is None or not image_dir.exists():
            raise FileNotFoundError(
                f"Could not find image directory for split "
                f"'{self.split}' in {self.root_dir}"
            )

        print(f"Using annotation file: {annotation_file}")
        print(f"Using image directory: {image_dir}")

        self.image_dir = image_dir

        # Load .mat file
        mat_data = sio.loadmat(str(annotation_file))

        # Parse annotations
        self.annotations = []

        # Try different possible structures
        if 'annotations' in mat_data:
            annos = mat_data['annotations'][0]
            # Debug: print available fields from first annotation
            if len(annos) > 0:
                print(f"Available fields in annotations: "
                      f"{annos[0].dtype.names}")

            for anno in annos:
                # Get image path
                if 'relative_im_path' in anno.dtype.names:
                    img_path = str(anno['relative_im_path'][0])
                elif 'fname' in anno.dtype.names:
                    img_path = str(anno['fname'][0])
                else:
                    # Try other possible field names
                    img_path = str(anno[0][0])

                # Get class ID - try different field names
                if 'class' in anno.dtype.names:
                    class_id = int(anno['class'][0][0]) - 1
                elif 'labels' in anno.dtype.names:
                    class_id = int(anno['labels'][0][0]) - 1
                else:
                    # Default to 0 if not found
                    class_id = 0

                # Get bounding box if available
                bbox = None
                if 'bbox_x1' in anno.dtype.names:
                    bbox = [
                        int(anno['bbox_x1'][0][0]),
                        int(anno['bbox_y1'][0][0]),
                        int(anno['bbox_x2'][0][0]),
                        int(anno['bbox_y2'][0][0])
                    ]

                self.annotations.append({
                    'image_path': img_path,
                    'class_id': class_id,
                    'bbox': bbox
                })
        else:
            # Alternative structure
            for key in mat_data.keys():
                if not key.startswith('__'):
                    print(f"Found key in .mat file: {key}")

        # Load class names
        meta_file = self.root_dir / 'devkit' / 'cars_meta.mat'
        if not meta_file.exists():
            # Search for meta file
            meta_files = list(self.root_dir.rglob('*meta*.mat'))
            if meta_files:
                meta_file = meta_files[0]

        if meta_file.exists():
            meta_data = sio.loadmat(str(meta_file))
            class_names_raw = meta_data['class_names'][0]
            self.class_names = [str(name[0]) for name in class_names_raw]
        else:
            print("Warning: Could not find class metadata file. "
                  "Using generic class names.")
            num_classes = max(
                [anno['class_id'] for anno in self.annotations]
            ) + 1
            self.class_names = [f"Class_{i}" for i in range(num_classes)]

        print(f"Loaded {len(self.annotations)} annotations "
              f"for {self.split} split")
        print(f"Number of classes: {len(self.class_names)}")

    def _use_label_mappings(self, label_mappings: Dict):
        """Use pre-built label mappings (for test split consistency)."""
        self.unique_brands = label_mappings['unique_brands']
        self.unique_models = label_mappings['unique_models']
        self.unique_years = label_mappings['unique_years']
        self.brand_to_idx = label_mappings['brand_to_idx']
        self.model_to_idx = label_mappings['model_to_idx']
        self.year_to_idx = label_mappings['year_to_idx']

        # Parse class names for this split
        self.brands = []
        self.models = []
        self.years = []
        for class_name in self.class_names:
            brand, model, year = self._parse_class_name(class_name)
            self.brands.append(brand)
            self.models.append(model)
            self.years.append(year)

        # Verify all labels exist in mappings (critical for test set!)
        missing_brands = set(self.brands) - set(self.unique_brands)
        missing_models = set(self.models) - set(self.unique_models)
        missing_years = set(self.years) - set(self.unique_years)

        if missing_brands:
            print(f"WARNING: Test has brands not in train: "
                  f"{missing_brands}")
        if missing_models:
            print(f"WARNING: Test has models not in train: "
                  f"{missing_models}")
        if missing_years:
            print(f"WARNING: Test set has years not in train: "
                  f"{missing_years}")

        print(f"✓ Using shared label mappings: "
              f"{len(self.unique_brands)} brands, "
              f"{len(self.unique_models)} models, "
              f"{len(self.unique_years)} years")
        print(f"✓ Test split has {len(set(self.brands))} unique brands, "
              f"{len(set(self.models))} unique models, "
              f"{len(set(self.years))} unique years")

    def _build_label_mappings(self):
        """Build mappings for brand, model, and year from class names."""
        self.brands = []
        self.models = []
        self.years = []

        # Parse each class name to extract brand, model, year
        for class_name in self.class_names:
            brand, model, year = self._parse_class_name(class_name)
            self.brands.append(brand)
            self.models.append(model)
            self.years.append(year)

        # Create unique mappings
        self.unique_brands = sorted(list(set(self.brands)))
        self.unique_models = sorted(list(set(self.models)))
        self.unique_years = sorted(list(set(self.years)))

        # Create label encoders
        self.brand_to_idx = {
            brand: idx for idx, brand in enumerate(self.unique_brands)
        }
        self.model_to_idx = {
            model: idx for idx, model in enumerate(self.unique_models)
        }
        self.year_to_idx = {
            year: idx for idx, year in enumerate(self.unique_years)
        }

        print(f"Unique brands: {len(self.unique_brands)}")
        print(f"Unique models: {len(self.unique_models)}")
        print(f"Unique years: {len(self.unique_years)}")
        print(f"Sample brands: {self.unique_brands[:5]}")
        print(f"Sample models: {self.unique_models[:5]}")
        print(f"Sample years: {self.unique_years[:5]}")

    def _parse_class_name(self, class_name: str) -> Tuple[str, str, int]:
        """
        Parse class name to extract brand, model, and year.

        Format: "Brand Model Year" (e.g., "AM General Hummer SUV 2000")

        Returns:
            Tuple of (brand, model, year)
        """
        # Extract year (last 4 digits)
        year_match = re.search(r'(\d{4})$', class_name.strip())
        if year_match:
            year = int(year_match.group(1))
            name_without_year = class_name[:year_match.start()].strip()
        else:
            year = 0  # Unknown year
            name_without_year = class_name.strip()

        # Split remaining into brand and model
        parts = name_without_year.split()

        if len(parts) == 0:
            brand = "Unknown"
            model = "Unknown"
        elif len(parts) == 1:
            brand = parts[0]
            model = "Unknown"
        else:
            # Common multi-word brands
            multi_word_brands = [
                'AM General', 'Aston Martin', 'Land Rover', 'Mercedes-Benz',
                'Rolls-Royce', 'Spyker C8'
            ]

            brand = None
            for mw_brand in multi_word_brands:
                if name_without_year.startswith(mw_brand):
                    brand = mw_brand
                    model = name_without_year[len(mw_brand):].strip()
                    break

            if brand is None:
                # Default: first word is brand, rest is model
                brand = parts[0]
                model = ' '.join(parts[1:])

        return brand, model, year

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Get item by index.

        Returns:
            Tuple of (image, labels) where labels is a dict containing:
                - 'brand': brand label index
                - 'model': model label index
                - 'year': year label index
                - 'class_id': original class id
        """
        anno = self.annotations[idx]

        # Load image - try multiple path resolution strategies
        img_path = self.image_dir / anno['image_path']

        if not img_path.exists():
            # Try just the filename
            img_path = self.image_dir / Path(anno['image_path']).name

        if not img_path.exists():
            # Try searching in subdirectories
            filename = Path(anno['image_path']).name
            possible_paths = list(self.image_dir.rglob(filename))
            if possible_paths:
                img_path = possible_paths[0]

        if not img_path.exists():
            # Try parent directory
            img_path = self.image_dir.parent / anno['image_path']

        if not img_path.exists():
            # Try root directory with just filename
            img_path = self.root_dir / Path(anno['image_path']).name

        if not img_path.exists():
            # Search entire root directory
            filename = Path(anno['image_path']).name
            possible_paths = list(self.root_dir.rglob(filename))
            if possible_paths:
                img_path = possible_paths[0]

        if not img_path.exists():
            raise FileNotFoundError(
                f"Could not find image: {anno['image_path']}\n"
                f"Tried paths:\n"
                f"  - {self.image_dir / anno['image_path']}\n"
                f"  - {self.image_dir / Path(anno['image_path']).name}\n"
                f"  - Searched in subdirectories of {self.image_dir}\n"
                f"  - {self.image_dir.parent / anno['image_path']}\n"
                f"  - Searched in {self.root_dir}"
            )

        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get labels
        class_id = anno['class_id']
        brand = self.brands[class_id]
        model = self.models[class_id]
        year = self.years[class_id]

        labels = {
            'brand': self.brand_to_idx[brand],
            'model': self.model_to_idx[model],
            'year': self.year_to_idx[year],
            'class_id': class_id,
            'brand_name': brand,
            'model_name': model,
            'year_value': year
        }

        return image, labels

    def get_num_classes(self) -> Dict[str, int]:
        """Get the number of classes for each label type."""
        return {
            'brand': len(self.unique_brands),
            'model': len(self.unique_models),
            'year': len(self.unique_years),
            'total': len(self.class_names)
        }

    def get_label_mappings(self) -> Dict:
        """Get label mappings to share with other splits."""
        return {
            'unique_brands': self.unique_brands,
            'unique_models': self.unique_models,
            'unique_years': self.unique_years,
            'brand_to_idx': self.brand_to_idx,
            'model_to_idx': self.model_to_idx,
            'year_to_idx': self.year_to_idx
        }


def get_default_transforms(
        img_size: int = 224,
        is_training: bool = True
) -> transforms.Compose:
    """
    Get default image transforms for Cars196 dataset.
    Args:
        img_size: Target image size
        is_training: Whether this is for training (includes augmentation)
    Returns:
        torchvision transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    auto_download: bool = True,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for Cars196 dataset.

    Args:
        root_dir: Root directory of dataset (None to auto-download)
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers for data loading
        auto_download: Automatically download from Kaggle
        cache_dir: Directory to cache downloaded dataset

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_transform = get_default_transforms(img_size, is_training=True)
    test_transform = get_default_transforms(img_size, is_training=False)

    train_dataset = Cars196Dataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        auto_download_kaggle=auto_download,
        cache_dir=cache_dir
    )

    # Use same root_dir from train dataset if auto-downloaded
    if root_dir is None and auto_download:
        root_dir = train_dataset.root_dir

    # Get label mappings from train dataset to ensure consistency
    label_mappings = train_dataset.get_label_mappings()
    print("\n✓ Sharing label mappings from train to test split...")

    test_dataset = Cars196Dataset(
        root_dir=root_dir,
        split='test',
        transform=test_transform,
        auto_download_kaggle=False,  # Already downloaded
        label_mappings=label_mappings  # Use same mappings as train!
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\nDataset Info:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")

    return train_loader, test_loader
