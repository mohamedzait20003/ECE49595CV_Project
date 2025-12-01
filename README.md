# Vision Transformer for Car Classification

A PyTorch implementation of Vision Transformer (ViT) for multi-label car classification using the Stanford Cars196 dataset. The model simultaneously predicts three attributes: **Brand**, **Model**, and **Year**.

## 🚗 Project Overview

This project implements a three-headed Vision Transformer classifier that can identify:
- **Car Brand** (e.g., BMW, Audi, Toyota)
- **Car Model** (e.g., M3, A4, Camry)
- **Manufacturing Year** (e.g., 2000, 2010, 2020)

### Model Architecture
- **Base Model**: Vision Transformer (ViT-Base)
- **Embedding Dimension**: 768
- **Number of Layers**: 12 transformer blocks
- **Attention Heads**: 12 multi-head self-attention
- **Patch Size**: 16×16
- **Image Size**: 224×224

## 📁 Project Structure

```
Codebase/
├── Model/
│   └── Model.py              # Vision Transformer implementation
├── Utilities/
│   └── Cars196.py            # Dataset loader with Kaggle integration
├── Notebooks/
│   └── notebook.ipynb        # Training and evaluation notebook
├── cache/                    # Dataset cache (auto-created)
├── checkpoints/              # Model checkpoints (auto-created)
├── results/                  # Training results & visualizations (auto-created)
├── main.py                   # Main training script
├── README.md                 # This file
└── .gitignore
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Kaggle account (for dataset access)

### Dependencies

```bash
pip install torch torchvision
pip install kagglehub scipy pillow tqdm matplotlib numpy
```

### Dataset Setup

The Stanford Cars196 dataset will be automatically downloaded from Kaggle on first run. You need to:

1. **Set up Kaggle credentials**:
   - Go to your Kaggle account settings
   - Create a new API token (downloads `kaggle.json`)
   - Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

2. The dataset will be cached in the `cache/` directory automatically.

## 🚀 Quick Start

### Option 1: Using Jupyter Notebook (Recommended)

1. Open `Notebooks/notebook.ipynb`
2. Run all cells sequentially
3. The notebook includes:
   - Dataset loading and visualization
   - Model initialization and architecture overview
   - Training with progress tracking
   - Evaluation metrics and visualizations
   - Sample predictions and inference examples

### Option 2: Using Python Script

```bash
python main.py
```

### Option 3: Training on Kaggle

1. Upload the following files to a new Kaggle notebook:
   - `Model/Model.py`
   - `Utilities/Cars196.py`
   - `Notebooks/notebook.ipynb`

2. Enable GPU accelerator in notebook settings

3. Run the notebook cells

## 📊 Training Configuration

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

### Training Pipeline
- ✅ **Automatic Dataset Download**: Kaggle integration with caching
- ✅ **Multi-Label Classification**: Three-headed classifier architecture
- ✅ **Early Stopping**: Prevents overfitting with patience-based stopping
- ✅ **Checkpointing**: Regular model saves and best model tracking
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau scheduler
- ✅ **Progress Tracking**: Real-time metrics with tqdm
- ✅ **Comprehensive Metrics**: Top-1, Top-5 accuracy for each task

### Evaluation & Visualization
- Training curves (loss and accuracy over epochs)
- Per-task performance breakdown (brand, model, year)
- Sample predictions with confidence scores
- Confusion matrix support

## 🎯 Model Variants

The codebase includes three pre-configured ViT variants:

| Model | Embed Dim | Depth | Heads | Parameters |
|-------|-----------|-------|-------|------------|
| **ViT-Small** | 384 | 12 | 6 | ~22M |
| **ViT-Base** | 768 | 12 | 12 | ~86M |
| **ViT-Large** | 1024 | 24 | 16 | ~307M |

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

## 💾 Outputs

After training, you'll find:

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
```

## 🔍 Inference Example

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
```

## 📚 Dataset Information

**Stanford Cars196 Dataset**
- **Total Classes**: 196 car classes
- **Training Samples**: ~8,144 images
- **Test Samples**: ~8,041 images
- **Source**: [Kaggle - Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)

The dataset includes fine-grained car classifications with annotations for make, model, and year.

## 🎓 Course Information

**ECE 49595 - Computer Vision**  
Purdue University  
Fall Semester 2025

## 📝 License

This project is for educational purposes as part of ECE 49595 coursework.

## 🙏 Acknowledgments

- Stanford AI Lab for the Cars196 dataset
- PyTorch team for the deep learning framework
- Vision Transformer paper: ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)

## 📧 Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.

---

**Happy Training! 🚀**
