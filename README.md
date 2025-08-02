# 🌾 Crop Insect Detection with YOLOv8 🐛

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive YOLOv8-based solution for detecting crop insects in agricultural images. This project provides end-to-end functionality for training, validating, and deploying object detection models specifically optimized for crop pest identification.

## 🎯 Features

- **🔥 YOLOv8 Integration**: Latest YOLO architecture for superior detection performance
- **📊 Smart Data Management**: Automated dataset organization and train/validation splitting
- **🧠 Multiple Training Strategies**: Basic, advanced, and specialized training approaches
- **📈 Comprehensive Monitoring**: Training metrics, plots, and model evaluation
- **⚡ Quick Testing**: Fast validation setup to verify configurations
- **🎛️ Flexible Configuration**: Easy parameter tuning and hyperparameter optimization
- **📱 Production Ready**: Exportable models for deployment in various formats

## 📋 Table of Contents

- [Dataset Information](#-dataset-information)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Training Options](#-training-options)
- [Usage Examples](#-usage-examples)
- [Model Performance](#-model-performance)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 📊 Dataset Information

### Overview
- **Dataset Size**: 50 annotated images
- **Classes**: 1 (Crop_Insect)
- **Format**: YOLO bounding box annotations
- **Split**: 40 training / 10 validation images (80/20)
- **Source**: CVAT exported in YOLO 1.1 format

### Data Structure
```
├── images/
│   ├── train/          # 40 training images (.jpg)
│   └── val/            # 10 validation images (.jpg)
└── labels/
    ├── train/          # 40 training annotations (.txt)
    └── val/            # 10 validation annotations (.txt)
```

### Annotation Format
Each annotation file contains bounding boxes in YOLO format:
```
class_id center_x center_y width height
```
Example: `0 0.495664 0.568516 0.381016 0.352500`

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (for cloning)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Crop_Insect_Annotation
   ```

2. **Install dependencies**
   ```bash
   pip install ultralytics
   ```

3. **Verify installation**
   ```bash
   python3 -c "from ultralytics import YOLO; print('✅ Installation successful!')"
   ```

## ⚡ Quick Start

### 1. Basic Training
```bash
python3 train_yolov8.py
```
Then select from the interactive menu:
- **Option 1**: Quick test (1 epoch) - Verify setup
- **Option 2**: Detection training (100 epochs) - Full training
- **Option 3**: Segmentation training (100 epochs) - Advanced detection

### 2. Advanced Training
```bash
python3 train_yolov8_improved.py
```
Choose from optimized strategies:
- **Strategy 1**: Improved training (recommended)
- **Strategy 2**: Nano model training (fastest)
- **Strategy 3**: No early stopping (full analysis)

### 3. Data Management
```bash
python3 split_data.py
```
Re-organize train/validation split with different ratios.

## 📁 Project Structure

```
Crop_Insect_Annotation/
├── 📊 DATASET
│   ├── images/
│   │   ├── train/           # 40 .jpg files
│   │   └── val/             # 10 .jpg files
│   └── labels/
│       ├── train/           # 40 .txt files
│       ├── val/             # 10 .txt files
│       └── train.cache      # YOLOv8 cache
│
├── 🧠 TRAINING SCRIPTS
│   ├── train_yolov8.py                # Basic training
│   ├── train_yolov8_improved.py       # Advanced training
│   └── split_data.py                  # Data management
│
├── ⚙️ CONFIGURATION
│   ├── data.yaml                      # YOLOv8 config
│   ├── TRAINING_IMPROVEMENTS.md       # Documentation
│   └── .gitignore                     # Git rules
│
├── 📈 TRAINING RESULTS
│   └── runs/
│       └── detect/
│           ├── setup_test/             # Initial test
│           ├── crop_insect_detection/  # Main training
│           └── crop_insect_test*/      # Test runs
│               └── weights/
│                   ├── best.pt        # Best model
│                   └── last.pt        # Latest model
│
├── 🏋️ PRE-TRAINED MODELS
│   ├── yolov8n.pt          # Nano (6.5 MB)
│   └── yolov8s.pt          # Small (22.6 MB)
│
└── 📜 LEGACY (Original CVAT Export)
    ├── obj_train_data/     # 50 original .txt files
    ├── obj.data           # Original config
    ├── obj.names          # Class names
    └── train.txt          # File list
```

## 🎯 Training Options

### Basic Training (`train_yolov8.py`)
Simple, interactive training with three modes:

**Quick Test (1 epoch)**
- Model: YOLOv8n
- Purpose: Verify setup and configuration
- Time: ~1-2 minutes

**Detection Training (100 epochs)**
- Model: YOLOv8s
- Batch size: 16
- Early stopping: 20 epochs patience
- Time: ~30-60 minutes

**Segmentation Training (100 epochs)**
- Model: YOLOv8s-seg
- Advanced pixel-level detection
- Time: ~45-90 minutes

### Advanced Training (`train_yolov8_improved.py`)
Optimized training with three strategies:

**Strategy 1: Improved Training (Recommended)**
```python
# Configuration
model = YOLOv8n
batch_size = 4
learning_rate = 0.001
patience = 50
optimizer = AdamW
augmentation = moderate
```

**Strategy 2: Nano Model Training**
```python
# Configuration  
model = YOLOv8n
image_size = 416
batch_size = 8
patience = 100
optimizer = SGD
augmentation = conservative
```

**Strategy 3: No Early Stopping**
```python
# Configuration
epochs = 100 (fixed)
early_stopping = disabled
purpose = full_learning_curve_analysis
```

## 💻 Usage Examples

### Training a Detection Model
```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='crop_insect_detection'
)
```

### Running Inference
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/crop_insect_detection/weights/best.pt')

# Run inference
results = model('path/to/new/image.jpg')

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    print(f"Detected {len(boxes)} insects")
```

### Model Export
```python
# Export to different formats
model.export(format='onnx')      # ONNX format
model.export(format='torchscript') # TorchScript
model.export(format='coreml')    # CoreML (iOS)
```

## 📊 Model Performance

### Training Results
Based on completed training runs:

| Model | mAP50 | mAP50-95 | Training Time | Model Size |
|-------|-------|----------|---------------|------------|
| YOLOv8n | 0.752 | 0.474 | ~20 min | 6.5 MB |
| YOLOv8s | 0.8+ | 0.5+ | ~45 min | 22.6 MB |

### Key Metrics
- **Precision**: Percentage of correct positive predictions
- **Recall**: Percentage of actual positives correctly identified
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: mAP averaged over IoU thresholds 0.5-0.95

## 🔧 Advanced Features

### Data Augmentation
The training pipeline includes sophisticated augmentation:
- **Rotation**: 5-15 degrees
- **Translation**: 5-10%
- **Scale**: 10-30%
- **HSV adjustments**: Color variation
- **Horizontal flip**: 50% probability
- **Mosaic**: Creates synthetic training examples

### Hyperparameter Optimization
Key parameters you can tune:

```yaml
# In data.yaml or training scripts
learning_rate: 0.001      # Lower for stable training
batch_size: 4-16          # Adjust based on GPU memory
patience: 20-100          # Early stopping patience
image_size: 416-640       # Input image resolution
augmentation: low/med/high # Data augmentation level
```

### Monitoring Training
Training progress is automatically logged to:
- **TensorBoard**: Real-time metrics visualization
- **Weights & Biases**: Advanced experiment tracking
- **Local files**: CSV logs and plots in `runs/` directory

## 🔍 Troubleshooting

### Common Issues

**Training stops early (around epoch 20-23)**
```bash
# Solutions:
# 1. Increase patience
python3 train_yolov8_improved.py  # Use Strategy 1

# 2. Reduce learning rate
# Edit training script: lr0=0.0005

# 3. Smaller batch size
# Edit training script: batch=4
```

**Memory errors**
```bash
# Reduce batch size
batch=4  # or even batch=2

# Reduce image size
imgsz=416  # instead of 640
```

**Poor performance**
```bash
# 1. Check data quality
python3 -c "from ultralytics.utils import yaml_load; print(yaml_load('data.yaml'))"

# 2. Increase training time
epochs=200  # More epochs

# 3. Add more data
# Collect additional annotated images
```

### Validation Steps
```bash
# 1. Verify dataset
ls images/train/ | wc -l  # Should show 40
ls labels/train/ | wc -l  # Should show 40

# 2. Test configuration
python3 -c "from ultralytics import YOLO; YOLO().train(data='data.yaml', epochs=1)"

# 3. Check file permissions
chmod +x *.py
```

## 🔬 Technical Details

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and training data
- **GPU**: Optional but recommended (CUDA-compatible)
- **OS**: macOS, Linux, or Windows

### Dependencies
```txt
ultralytics>=8.0.0    # YOLOv8 framework
torch>=1.8.0          # PyTorch backend
torchvision>=0.9.0    # Computer vision utilities
opencv-python>=4.6.0  # Image processing
pillow>=7.1.2         # Image handling
numpy>=1.18.0         # Numerical computing
matplotlib>=3.3.0     # Plotting and visualization
```

### Configuration Files

**data.yaml**
```yaml
# Dataset configuration
path: /path/to/Crop_Insect_Annotation
train: images/train
val: images/val
nc: 1
names:
  0: Crop_Insect
```

## 📈 Results and Outputs

### Training Outputs
After training completion, you'll find:

```
runs/detect/[experiment_name]/
├── weights/
│   ├── best.pt          # Best performing model
│   └── last.pt          # Final epoch model
├── train_batch*.jpg     # Training batch visualizations
├── val_batch*.jpg       # Validation batch visualizations
├── confusion_matrix.png # Model performance analysis
├── F1_curve.png        # F1 score across thresholds
├── P_curve.png         # Precision curve
├── R_curve.png         # Recall curve
├── PR_curve.png        # Precision-Recall curve
├── results.png         # Training metrics plot
└── args.yaml           # Training configuration
```

### Model Usage
```python
# Load trained model
model = YOLO('runs/detect/crop_insect_detection/weights/best.pt')

# Inference on new images
results = model('new_image.jpg', save=True, conf=0.5)

# Batch processing
results = model(['img1.jpg', 'img2.jpg'], save=True)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include type hints where possible
- Update documentation for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLOv8 framework
- **CVAT**: For annotation tools and export functionality
- **PyTorch**: For the underlying deep learning framework
- **Community**: For contributions and feedback

## 📞 Support

If you encounter issues or have questions:

1. **Check the troubleshooting section** above
2. **Search existing issues** in the repository
3. **Create a new issue** with detailed information:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

## 🚀 Future Enhancements

- [ ] **Multi-class detection**: Support for multiple insect types
- [ ] **Real-time inference**: Video stream processing
- [ ] **Mobile deployment**: TensorFlow Lite conversion
- [ ] **Web interface**: Browser-based detection tool
- [ ] **API integration**: RESTful API for remote inference
- [ ] **Data augmentation**: Advanced synthetic data generation

---

**Happy detecting! 🌾🐛**

For more information about YOLOv8, visit the [official documentation](https://docs.ultralytics.com/).