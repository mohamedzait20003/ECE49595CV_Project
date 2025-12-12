# Computer Vision Project: CNN vs Vision Transformer Comparison

A comprehensive comparative study of CNN and Vision Transformer architectures for car classification using the Stanford Cars196 dataset. This project includes:
1. **CNN Ablation Study**: ResNet-50 fine-tuning with various background augmentation strategies
2. **Vision Transformer Study**: Multi-label classification (Brand, Model, Year) using ViT architecture

## 🚗 Project Overview

### CNN Ablation Study
Evaluates the robustness of ResNet-50 to different background conditions:
- **Baseline**: Standard fine-tuning with data augmentation
- **Background Manipulations**: Blur, noise, pixelation, white-out, and crop variants
- **Goal**: Determine if background removal/manipulation improves model focus on car features

### Vision Transformer Study
Implements a three-headed ViT classifier that simultaneously identifies:
- **Car Brand** (e.g., BMW, Audi, Toyota)
- **Car Model** (e.g., M3, A4, Camry)
- **Manufacturing Year** (e.g., 2000, 2010, 2020)

#### ViT Architecture Details
- **Base Model**: Vision Transformer (ViT-Base)
- **Embedding Dimension**: 768
- **Number of Layers**: 12 transformer blocks
- **Attention Heads**: 12 multi-head self-attention
- **Patch Size**: 16×16
- **Image Size**: 224×224

## 📁 Project Structure

```
Codebase/
├── CNN Ablation Study/
│   ├── ablation/
│   │   ├── finetune_cars_resnet50.py          # Baseline ResNet-50 training
│   │   ├── finetune_cars_resnet50_noaug.py    # No augmentation variant
│   │   ├── blur.py                             # Blur background augmentation
│   │   ├── noise.py                            # Noise background augmentation
│   │   ├── pixel.py                            # Pixelation background augmentation
│   │   ├── white.py                            # White background augmentation
│   │   ├── crop.py                             # Crop-based augmentation
│   │   ├── crop(distort).py                    # Distorted crop variant
│   │   └── crop copy.py                        # Alternative crop implementation
│   ├── clean/
│   │   ├── count.py                            # Dataset statistics utility
│   │   ├── delete.py                           # Data cleanup utility
│   │   └── missing_indices.json                # Missing data tracking
│   ├── results/
│   │   ├── baseline.txt                        # Baseline training results
│   │   ├── blur.txt                            # Blur augmentation results
│   │   ├── crop.txt                            # Crop augmentation results
│   │   ├── noise.txt                           # Noise augmentation results
│   │   ├── pixelate.txt                        # Pixelation results
│   │   └── white.txt                           # White background results
│   ├── cars_test_annos_withlabels.mat         # Test annotations
│   └── acknowlegements.txt                     # Dataset acknowledgments
│
├── VIT Study/
│   ├── Model/
│   │   └── Model.py              # Vision Transformer implementation
│   ├── Utilities/
│   │   └── Cars196.py            # Dataset loader with Kaggle integration
│   ├── Notebooks/
│   │   └── notebook.ipynb        # Training and evaluation notebook
│   └── main.py                   # Main training script
│
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Kaggle account (for dataset access via VIT Study)
- PyTorch 1.9+

### Dependencies

#### For CNN Ablation Study:
```bash
pip install torch torchvision
pip install scipy pillow numpy tqdm scikit-learn
```

#### For Vision Transformer Study:
```bash
pip install torch torchvision
pip install kagglehub scipy pillow tqdm matplotlib numpy
```

### Dataset Setup

#### CNN Ablation Study
The Stanford Cars dataset must be manually downloaded and placed in the appropriate directory structure. The scripts expect:
- `devkit/cars_meta.mat` for class names
- `devkit/cars_train_annos.mat` for training annotations
- `cars_test_annos_withlabels.mat` for test annotations (included in repo)
- Training images in `cars_train/`
- Test images in `cars_test/`

#### Vision Transformer Study
The dataset will be automatically downloaded from Kaggle on first run:

1. **Set up Kaggle credentials**:
   - Go to your Kaggle account settings
   - Create a new API token (downloads `kaggle.json`)
   - Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

2. The dataset will be cached in the `cache/` directory automatically.

## 🚀 Quick Start

### CNN Ablation Study

#### 1. Baseline Training (with augmentation)
```bash
cd "CNN Ablation Study"
python ablation/finetune_cars_resnet50.py --data_root /path/to/stanford_cars --output_dir ./outputs/baseline --epochs 50 --batch_size 32 --lr 1e-4
```

#### 2. No Augmentation Training
```bash
python ablation/finetune_cars_resnet50_noaug.py --data_root /path/to/stanford_cars --output_dir ./outputs/noaug --epochs 50 --batch_size 32 --lr 1e-4
```

#### 3. Background Manipulation Experiments
Run each ablation variant:

**Blur Background:**
```bash
python ablation/blur.py --data_root /path/to/stanford_cars --output_dir ./outputs/blur --epochs 50 --batch_size 32 --lr 1e-4
```

**Noise Background:**
```bash
python ablation/noise.py --data_root /path/to/stanford_cars --output_dir ./outputs/noise --epochs 50 --batch_size 32 --lr 1e-4
```

**Pixelate Background:**
```bash
python ablation/pixel.py --data_root /path/to/stanford_cars --output_dir ./outputs/pixelate --epochs 50 --batch_size 32 --lr 1e-4
```

**White Background:**
```bash
python ablation/white.py --data_root /path/to/stanford_cars --output_dir ./outputs/white --epochs 50 --batch_size 32 --lr 1e-4
```

**Crop-based Augmentation:**
```bash
python ablation/crop.py --data_root /path/to/stanford_cars --output_dir ./outputs/crop --epochs 50 --batch_size 32 --lr 1e-4
```

### Vision Transformer Study

#### Option 1: Using Jupyter Notebook (Recommended)
```bash
cd "VIT Study"
jupyter notebook Notebooks/notebook.ipynb
```

1. Open `Notebooks/notebook.ipynb`
2. Run all cells sequentially
3. The notebook includes:
   - Dataset loading and visualization
   - Model initialization and architecture overview
   - Training with progress tracking
   - Evaluation metrics and visualizations
   - Sample predictions and inference examples

#### Option 2: Using Python Script
```bash
cd "VIT Study"
python main.py
```

#### Option 3: Training on Kaggle
1. Upload the following files to a new Kaggle notebook:
   - `Model/Model.py`
   - `Utilities/Cars196.py`
   - `Notebooks/notebook.ipynb`

2. Enable GPU accelerator in notebook settings

3. Run the notebook cells

## 📊 Training Configuration

### CNN Ablation Study (ResNet-50)

Default hyperparameters used across all ablation experiments:

```python
{
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 50,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'patience': 5,
    'img_size': 224,
    
    # Data augmentation (baseline & variants)
    'random_resized_crop': (224, 224),
    'random_horizontal_flip': True,
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}
```

**Ablation Variants:**
- **Baseline**: Standard training with RandomResizedCrop and horizontal flip
- **No Augmentation**: Only resize and normalize
- **Blur**: Gaussian blur applied to background regions
- **Noise**: Gaussian noise added to background regions
- **Pixelate**: Background pixelation effect
- **White**: Replace background with white color
- **Crop**: Bbox-based cropping with padding

### Vision Transformer Study

Default hyperparameters in `notebook.ipynb`:

```python
config = {
    'batch_size': 32,
    'img_size': 224,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience': 10,           # Early stopping
    'save_frequency': 5,      # Checkpoint every N epochs
    
    # Model architecture
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
}
```

## 📈 Features

### CNN Ablation Study
- ✅ **Multiple Background Strategies**: Systematic evaluation of background manipulation effects
- ✅ **Baseline Comparison**: Controlled experiments with consistent hyperparameters
- ✅ **ResNet-50 Architecture**: Pre-trained ImageNet weights with fine-tuning
- ✅ **Comprehensive Logging**: Training history saved for each variant
- ✅ **Result Tracking**: JSON-formatted results for easy comparison
- ✅ **Bbox Annotations**: Utilizes bounding box information for crop variants
- ✅ **Early Stopping**: Patience-based training termination
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau scheduler

### Vision Transformer Study

#### Training Pipeline
- ✅ **Automatic Dataset Download**: Kaggle integration with caching
- ✅ **Multi-Label Classification**: Three-headed classifier architecture
- ✅ **Early Stopping**: Prevents overfitting with patience-based stopping
- ✅ **Checkpointing**: Regular model saves and best model tracking
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau scheduler
- ✅ **Progress Tracking**: Real-time metrics with tqdm
- ✅ **Comprehensive Metrics**: Top-1, Top-5 accuracy for each task

#### Evaluation & Visualization
- Training curves (loss and accuracy over epochs)
- Per-task performance breakdown (brand, model, year)
- Sample predictions with confidence scores
- Confusion matrix support

## 🎯 Model Variants (Vision Transformer Study)

The VIT Study includes three pre-configured ViT variants:

| Model | Embed Dim | Depth | Heads | Parameters | Use Case |
|-------|-----------|-------|-------|------------|----------|
| **ViT-Small** | 384 | 12 | 6 | ~22M | Fast prototyping, limited GPU |
| **ViT-Base** | 768 | 12 | 12 | ~86M | Balanced performance (default) |
| **ViT-Large** | 1024 | 24 | 16 | ~307M | Maximum accuracy, high compute |

Create different variants:
```python
from Model.Model import create_vit_small, create_vit_base, create_vit_large

# Small model (faster training)
model = create_vit_small(num_classes_brand, num_classes_model, num_classes_year)

# Base model (default)
model = create_vit_base(num_classes_brand, num_classes_model, num_classes_year)

# Large model (best performance)
model = create_vit_large(num_classes_brand, num_classes_model, num_classes_year)
```

## 🧪 Research Questions

This project investigates:

1. **CNN Background Sensitivity**: Does background manipulation (blur, noise, crop, etc.) improve or hinder ResNet-50's ability to focus on car features?

2. **Augmentation Impact**: How does data augmentation affect model generalization on fine-grained car classification?

3. **CNN vs Transformer**: How do traditional CNNs (ResNet-50) compare to Vision Transformers on the same dataset?

4. **Multi-label Learning**: Can ViT effectively decompose car classification into multiple attributes (brand, model, year)? 
   - ⚠️ **Status**: Multi-label approach encountered classification challenges; further investigation needed

## ⚠️ Known Issues & Limitations

### Vision Transformer Study
- **Multi-label Classification Failure**: The current three-headed ViT implementation for Brand/Model/Year prediction has shown poor classification performance
- **Possible Root Causes**:
  - Label extraction from 196-class structure may be incorrect or inconsistent
  - Insufficient training data for three separate classification heads
  - Class imbalance issues across brand/model/year categories
  - Training from scratch (no pre-trained weights) may be insufficient
- **Recommended Next Steps**:
  - Debug label extraction and verify brand/model/year mappings
  - Consider using standard 196-class classification (single head)
  - Explore pre-trained ViT models and fine-tuning
  - Implement proper class weighting or focal loss for imbalanced classes

### CNN Ablation Study
- Background manipulation results may vary based on specific car models and image quality
- Some augmentation strategies may introduce artifacts that affect performance

## 🛠️ Utility Scripts

### CNN Ablation Study Utilities

Located in `CNN Ablation Study/clean/`:

- **count.py**: Calculate dataset statistics and class distributions
- **delete.py**: Clean up incomplete or corrupted data
- **missing_indices.json**: Track missing or problematic samples

## 💾 Outputs

### CNN Ablation Study

After training each variant, results are saved in `CNN Ablation Study/results/`:

```
results/
├── baseline.txt         # Baseline training metrics (epoch, loss, accuracy)
├── blur.txt            # Blur background results
├── crop.txt            # Crop augmentation results
├── noise.txt           # Noise background results
├── pixelate.txt        # Pixelation results
└── white.txt           # White background results
```

Each result file contains JSON-formatted training history:
```json
[
  {
    "epoch": 1,
    "train_loss": 4.321,
    "train_acc": 0.108,
    "val_loss": 4.048,
    "val_acc": 0.128
  },
  ...
]
```

### Vision Transformer Study

After training, outputs are saved in `VIT Study/`:

```
checkpoints/
├── best_model.pth           # Best model based on validation loss
├── final_model.pth          # Final epoch model
└── checkpoint_epoch_*.pth   # Periodic checkpoints

results/
├── training_history.json    # Training metrics per epoch
├── final_results.json       # Final test set performance
├── training_curves.png      # Loss and accuracy plots
└── final_performance.png    # Performance bar charts

cache/
└── stanford-cars/           # Cached dataset (auto-downloaded)
```

## 🔍 Usage Examples

### CNN Ablation Study - Inference

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load trained model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 196)  # 196 car classes
checkpoint = torch.load('outputs/baseline/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('car_image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted class: {predicted.item()}")
```

### Vision Transformer Study - Inference

```python
from Model.Model import create_vit_base
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load trained model
model = create_vit_base(num_classes_brand, num_classes_model, num_classes_year)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('car_image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    brand_out, model_out, year_out = model(image_tensor)
    brand_pred = brand_out.argmax(dim=1)
    model_pred = model_out.argmax(dim=1)
    year_pred = year_out.argmax(dim=1)
    
    print(f"Brand: {brand_pred.item()}")
    print(f"Model: {model_pred.item()}")
    print(f"Year: {year_pred.item()}")
```

## 📚 Dataset Information

**Stanford Cars196 Dataset**
- **Total Classes**: 196 car classes
- **Training Samples**: ~8,144 images
- **Test Samples**: ~8,041 images
- **Image Resolution**: Variable (resized to 224×224)
- **Annotations**: Bounding boxes, class labels
- **Source**: [Kaggle - Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)

The dataset includes fine-grained car classifications with annotations for make, model, and year. Both studies utilize this dataset:
- **CNN Study**: Uses standard 196-class classification
- **VIT Study**: Decomposes classes into multi-label Brand/Model/Year prediction

## 🔬 Experimental Results

### CNN Ablation Study - Key Findings

The ablation study evaluates how background manipulation affects model performance:

| Variant | Strategy | Purpose |
|---------|----------|---------|
| **Baseline** | Standard augmentation | Reference performance |
| **No Aug** | Minimal augmentation | Evaluate augmentation impact |
| **Blur** | Gaussian blur background | Reduce background detail |
| **Noise** | Add noise to background | Test robustness |
| **Pixelate** | Pixelate background | Remove fine background details |
| **White** | Replace background with white | Eliminate background completely |
| **Crop** | Bbox-based cropping | Focus on car region |

Results are stored in `CNN Ablation Study/results/` with detailed training metrics for comparison.

### Vision Transformer Study - Architecture Variants

⚠️ **Note**: The Vision Transformer multi-label classification approach encountered challenges and may require further optimization. Consider the following alternatives:

**Alternative Approaches:**
1. **Single-label Classification**: Use standard 196-class classification instead of multi-label decomposition
2. **Transfer Learning**: Initialize with pre-trained ViT weights (ImageNet-21k or ImageNet-1k)
3. **Simplified Architecture**: Start with ViT-Small for faster experimentation
4. **Hybrid Approach**: Use CNN backbone with transformer layers

The VIT study provides multiple model sizes for different compute budgets:

## 🎓 Course Information

**ECE 49595 - Computer Vision**  
Purdue University  
Fall Semester 2025

This project is part of the coursework for ECE 49595, exploring modern computer vision architectures and their application to fine-grained classification tasks.

## 🔧 Troubleshooting

### CNN Ablation Study

**Issue**: Dataset not found
- **Solution**: Ensure Stanford Cars dataset is downloaded and paths are correctly set in `--data_root`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size using `--batch_size 16` or `--batch_size 8`

**Issue**: Poor convergence
- **Solution**: Try different learning rates using `--lr 5e-5` or enable/disable augmentation

### Vision Transformer Study

**Issue**: Classification failure / Poor performance
- **Possible Causes**: 
  - Multi-label classification may be too challenging for the dataset size
  - Brand/Model/Year decomposition might not align well with the original 196 classes
  - Insufficient training epochs or learning rate issues
  - Class imbalance across brand/model/year labels
- **Solutions**: 
  - Consider using single-label classification (196 classes) instead of multi-label
  - Increase training epochs and adjust learning rate
  - Use data augmentation more aggressively
  - Try pre-trained ViT weights instead of training from scratch
  - Verify label extraction from dataset is correct

**Issue**: Kaggle API authentication failed
- **Solution**: Verify `kaggle.json` is in the correct location with proper permissions

**Issue**: Model training is too slow
- **Solution**: Use ViT-Small variant or reduce batch size, ensure GPU is being utilized

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt` (if available)

**Issue**: Model not converging
- **Solution**: Check learning rate (try 1e-3 to 1e-5 range), verify loss computation for multi-label setup, ensure proper weight initialization

## 📊 Comparing Results

To compare CNN and ViT performance:

1. Train both models on the same dataset splits
2. Compare accuracy, training time, and computational requirements
3. Analyze which backgrounds/augmentations help each architecture
4. Evaluate multi-label vs single-label classification approaches

Example comparison script (create as needed):
```python
import json

# Load CNN results
with open('CNN Ablation Study/results/baseline.txt', 'r') as f:
    cnn_results = json.load(f)

# Load ViT results
with open('VIT Study/results/final_results.json', 'r') as f:
    vit_results = json.load(f)

# Compare metrics
print("CNN Best Val Accuracy:", max([e['val_acc'] for e in cnn_results]))
print("ViT Best Val Accuracy:", vit_results['best_val_accuracy'])
```

## 📝 License

This project is for educational purposes as part of ECE 49595 coursework.

## 🙏 Acknowledgments

- Stanford AI Lab for the Cars196 dataset
- PyTorch team for the deep learning framework
- Vision Transformer paper: ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.
- ResNet paper: ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) by He et al.
- Kaggle for hosting the Stanford Cars dataset

## 📂 Additional Resources

- **Stanford Cars Dataset**: [Original Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- **Vision Transformer Paper**: [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- **ResNet Paper**: [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- **Fine-Grained Classification**: [Stanford Vision Lab](http://vision.stanford.edu/)

## 👥 Contributors

This project is developed as part of academic coursework for ECE 49595 at Purdue University.

## 📧 Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.

---

**Happy Training! 🚀**
