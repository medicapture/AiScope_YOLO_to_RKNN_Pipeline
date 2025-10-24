# aiScope YOLO to RKNN Conversion Pipeline

An integrated pipeline for converting YOLO models (YOLOv8, YOLO11) to RKNN format for deployment on Rockchip NPU platforms, specifically designed for **MediCapture aiScope** - an AI-powered feature for medical video recorders.

## What is aiScope?

**aiScope** is a built-in AI inference feature for MediCapture MVR/MTR series medical video recorders that enables real-time object detection and image classification during surgical procedures. This pipeline provides the essential toolchain to convert your custom YOLO models into the RKNN format required by aiScope.

### The "Last Mile" Problem - Solved

Researchers typically develop AI models on PCs with GPUs, but deploying them for **real-time surgical use** requires dedicated, robust hardware. aiScope solves this by providing:

- **Real-time inference**: 60fps at 1080p/2160p with <30ms latency
- **Certified medical device**: Built into Medical Class 1 certified video recorders
- **Dual camera support**: Run different models simultaneously on multiple video inputs
- **Live visualization**: Augmented video overlay with bounding boxes and labels
- **Data output**: JSON-formatted inference results via LAN for external devices
- **Low power**: ~2W power consumption for AI inference at the maximum load of 90 fps (on top of the MVR/MTR device power consumption of 10-13W). 

### Medical Applications

aiScope enables various Vision AI applications in medical contexts:

- **Detection**: Polyp detection, tumor detection, bleeding detection, surgical tools tracking, foreign object detection
- **Classification**: Tissue type identification, procedure phase recognition, image quality assessment
- **Tracking**: Multi-object tracking with velocity and lifetime metrics
- **Research**: Custom dataset creation, model evaluation, clinical validation

**This pipeline is the bridge** between your research models (PyTorch) and real-world deployment (aiScope/RKNN).

---

## Features

- **Fine-tuning Support**: Optional fine-tuning on custom datasets before conversion
- **Activation Replacement**: Automatically replaces SiLU/Swish activations with LeakyReLU for better RKNN compatibility
- **Pruned Model Support**: Handles structurally pruned modules (via torch-pruning)
- **Pretrained Models**: Pre-optimized models trained on COCO dataset for quick testing
- **Multi-task Support**: Works with detection and classification models
- **INT8 Quantization**: Automatic INT8 quantization for optimal inference speed
- **WSL2 Compatible**: Automatically detects and configures for WSL2 environments
- **aiScope Ready**: Output models are directly compatible with MediCapture aiScope devices

---

## aiScope Development Roadmap

**Complete workflow from research to clinical deployment:**

This pipeline handles **Steps 2-5** of the complete aiScope development process. Here's your roadmap:

| Step | Stage | What You'll Do | Time Required | Where |
|------|-------|----------------|---------------|-------|
| 1️⃣ | **Select Base Model** | Choose a pretrained YOLO model optimized for aiScope | 5 min | This repository |
| 2️⃣ | **Fine-tune Model** | Train on your custom medical dataset | 1-4 hours (can be longer due to dataset size and training settings) | **This pipeline** |
| 3️⃣ | **Validate Accuracy** | Test model performance on validation set | 10 min | **This pipeline** |
| 4️⃣ | **Convert to RKNN** | Export PyTorch → ONNX → RKNN with INT8 quantization | 2-5 min | **This pipeline** |
| 5️⃣ | **Package for aiScope** | Create deployment package with model + configuration | 5 min | aiScope documentation |
| 6️⃣ | **Upload to Device** | Transfer `.tar.gz` to MTR/MVR via USB | 2 min | aiScope device |
| 7️⃣ | **Evaluate & Deploy** | Test with free evaluation mode, then activate for production | 10-60 min | aiScope device |

**Total Time:**
- **Quick test** (no fine-tuning): ~15 minutes (Steps 1, 4, 5, 6, 7)
- **Full workflow** (with fine-tuning): ~2-6 hours (all steps)

**This pipeline automates Steps 2-4**, providing a seamless path from research to deployment.

---

## Quick Start

**Get started in 2 minutes:**

```bash
# Direct conversion (2 minutes)
python aiscope_pipeline.py \
    --model pretrained_models/yolo11s_unpruned.pt
```

Output: `pretrained_models/yolo11s_unpruned_i8.rknn` ready for aiScope deployment!

---

## Requirements

### Recommended System
- **OS**: Linux (or WSL2 on Windows)
- **Python**: 3.10
- **PyTorch**: 2.4.1 with CUDA 12.4 (for training/fine-tuning)
- **RKNN Toolkit2**: For ONNX → RKNN conversion
- **GPU**: CUDA-capable (optional, for fine-tuning)

### Target Deployment Platform
- MediCapture MTR133, MTR156, or MVR436 with firmware ≥250901
- Any Rockchip device with RK3562/3566/3568/3576/3588 NPU

---

## Installation

1. **Clone this repository:**
```bash
git clone https://github.com/medicapture/AiScope_YOLO_to_RKNN_Pipeline.git
cd AiScope_YOLO_to_RKNN_Pipeline
```

2. **Install PyTorch with CUDA** (skip if already installed):
```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

3. **Install remaining dependencies:**
```bash
pip install -r requirements.txt
pip install rknn-toolkit2 --no-deps
```

> **Note:** The `--no-deps` flag prevents pip from overriding your GPU-enabled PyTorch with a CPU-only version.

---

## Usage

### Basic Conversion (PT → ONNX → RKNN)

Convert a pretrained YOLO model directly to RKNN format:

```bash
python aiscope_pipeline.py \
    --model pretrained_models/yolo11s_unpruned.pt
```

**Output files:**
- `pretrained_models/yolo11s_unpruned.onnx` - Intermediate ONNX model
- `pretrained_models/yolo11s_unpruned_i8.rknn` - Final INT8 quantized RKNN model for aiScope

### Fine-tuning + Conversion

Fine-tune the model on your custom dataset before conversion:

```bash
python aiscope_pipeline.py \
    --model pretrained_models/yolo11s_unpruned.pt \
    --fine-tune
```

**Prerequisites for fine-tuning:**
1. Prepare dataset in YOLO format
2. Create dataset config YAML in `cfg/` (see `cfg/coco128.yaml` as example)
3. Update training parameters in `ultralytics/cfg/default.yaml`

**Note**: Before fine-tuning, update the `data` field in `ultralytics/cfg/default.yaml` to point to your dataset config.

### Custom Image Size

Adjust input resolution based on your video aspect ratio:

```bash
# For 16:9 video
python aiscope_pipeline.py \
    --model pretrained_models/yolo11s_unpruned.pt \
    --img-size 384 640

# For square video (1:1 aspect ratio)
python aiscope_pipeline.py \
    --model pretrained_models/yolov8s_unpruned.pt \
    --img-size 640
```

---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `pretrained_models/yolov8s_unpruned.pt` | Path to input model (`.pt` file) |
| `--fine-tune` | flag | False | Enable fine-tuning before conversion |
| `--img-size` | int/list | None | Image size for ONNX export (e.g., `640` or `384 640`) |
| `--quant-dataset` | str | `./quantization_dataset/dataset.txt` | Path to quantization dataset list |

---

## Pretrained Models

All models are available in the `pretrained_models/` folder, pre-optimized for RKNN deployment:

### Detection Models

#### YOLO11 (Latest)
- **yolo11s_unpruned.pt** - Small, fast, COCO-80 pretrained
- **yolo11m_unpruned.pt** - Medium, balanced performance
- **yolo11m_pruned.pt** - Medium, structurally pruned for speed

#### YOLOv8 (Stable)
- **yolov8s_unpruned.pt** - Small, fast, legacy baseline
- **yolov8m_unpruned.pt** - Medium, higher accuracy
- **yolov8m_pruned.pt** - Medium, structurally pruned

### Classification Models
- **yolo11m_cls.pt** - ImageNet-1000 pretrained classifier

---

## Pipeline Overview

The conversion pipeline consists of three main stages:

### 1. Optional Fine-tuning
- Loads the base PyTorch model
- Replaces all activations with LeakyReLU (RKNN-compatible)
- Detects if model is pruned (C2f_v2 modules)
- Trains on custom dataset using Ultralytics API
- Saves best weights to `runs/detect/{project_name}/weights/best.pt`

### 2. ONNX Export
- Exports PyTorch model to ONNX format
- For classification models: automatically removes Softmax layer
- Optimizes graph structure for RKNN conversion
- Outputs: `{model_name}.onnx`

### 3. RKNN Conversion
- Configures normalization (mean=[0,0,0], std=[255,255,255])
- Loads ONNX model
- Performs INT8 quantization using calibration dataset
- Builds and exports RKNN model
- Outputs: `{model_name}_i8.rknn`

---

## Dataset Preparation

### Training Dataset (for fine-tuning)

Organize your dataset in YOLO format:

```
your_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/
│       ├── img101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt  # YOLO format: class_id x_center y_center width height
    │   ├── img002.txt
    │   └── ...
    └── val/
        ├── img101.txt
        └── ...
```

Create a dataset config file (e.g., `cfg/my_dataset.yaml`):

```yaml
path: /absolute/path/to/your_dataset
train: images/train
val: images/val

names:
  0: polyp
  1: bleeding
  2: normal_tissue
  # ... add all your classes
```

Update `ultralytics/cfg/default.yaml`:

```yaml
data: ./cfg/my_dataset.yaml
epochs: 100
batch: 16
imgsz: 640
```

### Quantization Dataset (Required for Conversion)

The pipeline automatically performs INT8 quantization. Prepare 50-200 representative images from your dataset.

**Example `dataset.txt`:**
```
quantization_dataset/000000000001.jpg
quantization_dataset/000000000016.jpg
quantization_dataset/000000000019.jpg
...
```

**Important:** Use images representative of your deployment scenario (lighting, angles, object sizes, etc.)

---

## Deployment to aiScope

### Step 1: Convert Your Model

Use this pipeline to generate the `.rknn` file:

```bash
python aiscope_pipeline.py \
    --model pretrained_models/yolo11s_unpruned.pt \
    --fine-tune \
    --img-size 384 640
```

### Step 2: Package for aiScope

After conversion completes, you'll have:
- Fine-tuned model: `runs/detect/pipeline_test/weights/best_i8.rknn`
- Class names: Define in your dataset YAML
- Quantization images: Already used during conversion

### Step 3: Upload to MTR/MVR Device

1. Copy the `.rknn` file to a USB flash drive
2. Prepare a configuration file with class names and parameters
3. Package as `custom_name.tar.gz` (see aiScope documentation)
4. Upload to your MediCapture device
5. Assign the model to a video input

### Step 4: Evaluate on aiScope

**Free Evaluation Mode:**
- Test your model with 10-minute inference sessions
- Watermark will appear on output
- Verify inference accuracy with test objects

**Production Mode:**
- Purchase aiScope activation key to remove limitations
- Deploy for clinical use (following regulatory requirements)

**Performance Expectations:**
- Single 1080p input: 60 fps, ~19ms latency
- Single 2160p input: 60 fps, ~28ms latency
- Dual input: 30+30 fps or 60+30 fps (90 fps max total)

---

## Output Files

After running the pipeline, you'll find:

| File | Description |
|------|-------------|
| `{model_name}.onnx` | Intermediate ONNX model |
| `{model_name}_softmax_removed.onnx` | (Classification only) ONNX without Softmax layer |
| `{model_name}_i8.rknn` | **Final INT8 quantized RKNN model for aiScope deployment** |
| `model_conversion.log` | Detailed conversion logs with timestamps |
| `runs/detect/pipeline_test*/` | (Fine-tuning) Training results, plots, weights |

---

## Logging

The pipeline generates detailed colored logs for easy monitoring:

- **GREEN (INFO)** - Normal progress messages
- **YELLOW (WARNING)** - Warnings and important notices
- **RED (ERROR)** - Errors and failures

All logs are also saved to `model_conversion.log` for later review:

```bash
# Monitor in real-time
tail -f model_conversion.log

# Search for errors
grep ERROR model_conversion.log
```

---

## Advanced Configuration

### Training Hyperparameters

Edit `ultralytics/cfg/default.yaml` to customize training behavior:

```yaml
# Training duration
epochs: 100              # More epochs for larger datasets
patience: 50             # Early stopping patience

# Batch and image size
batch: 16                # Adjust based on GPU memory
imgsz: 640               # Match your deployment resolution

# Folder name
name: folder_name        # Select the name for the save folder

# Learning rates
lr0: 0.0002              # Initial learning rate
lrf: 0.4                 # Final learning rate multiplier

# Augmentation
mosaic: 1.0              # Mosaic augmentation probability
mixup: 0.15              # Mixup augmentation probability
copy_paste: 0.4          # Copy-paste augmentation (for detection)

# Loss weights
box: 7.5                 # Bounding box loss gain
cls: 0.5                 # Classification loss gain
dfl: 1.5                 # Distribution focal loss gain
```

### RKNN Quantization

The pipeline automatically performs INT8 quantization with these settings (hardcoded in `aiscope_pipeline.py:275-277`):

```python
mean_values=[[0, 0, 0]]       # No mean normalization
std_values=[[255, 255, 255]]   # Scale pixels to [0, 1]
```

**These values are optimized for aiScope.** Changing them may cause inference issues on the device.

---

## Regulatory and Clinical Use

### Important Disclaimer

This pipeline and the pretrained models are provided for **research and development purposes only**.

**For clinical deployment:**
- aiScope is a feature of Medical Class 1 certified video recorders (EN 60601-1)
- Using aiScope for **documentation or physician assistance** does not alter device classification
- Custom models should **not be used for diagnostic purposes** or to replace clinical decision-making
- For models intended for clinical diagnosis, follow the established **FDA approval process** for AI/ML medical devices

**Consult with regulatory experts** before deploying custom models in clinical settings.

---

## License

This project uses components from:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - AGPL-3.0 License
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2) - Rockchip License

**Commercial Use:** Ultralytics offers commercial licensing options. Contact Ultralytics for details.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{aiscope_yolo_to_rknn_pipeline,
  title = {aiScope YOLO to RKNN Conversion Pipeline},
  author = {JoeChen0123},
  year = {2025},
  url = {https://github.com/medicapture/AiScope_YOLO_to_RKNN_Pipeline}
}
```

**For aiScope-related publications**, also cite MediCapture aiScope documentation.

---

## Resources

### Documentation
- **aiScope**: Contact MediCapture for aiScope user manual
- **Ultralytics YOLO**: [https://docs.ultralytics.com](https://docs.ultralytics.com)
- **RKNN Toolkit2**: [https://github.com/airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)

### Support
- **Pipeline Issues**: [GitHub Issues](https://github.com/medicapture/AiScope_YOLO_to_RKNN_Pipeline/issues)
- **aiScope Support**: Contact MediCapture sales representative

### MediCapture
- **Website**: [https://medicapture.com](https://medicapture.com)
- **aiScope Devices**: MTR133, MTR156, MVR436
- **Firmware**: Requires version ≥250901

---

## Acknowledgments

- **Ultralytics Team** - For the excellent YOLO implementation and training framework
- **Rockchip** - For RKNN Toolkit2 and NPU hardware
- **MediCapture** - For aiScope platform and medical device integration
- **Open-source community** - For invaluable tools and libraries

---

## FAQ

### Can I use models other than YOLO?

**Answer:** Technically yes, but not recommended. This pipeline is optimized for YOLO architectures (Ultralytics). Using other architectures (e.g., Faster R-CNN, EfficientDet) requires:
- Custom ONNX export logic
- Unknown RKNN compatibility
- Unpredictable performance on aiScope

YOLO is widely recognized for efficiency and speed in real-time video processing, making it ideal for medical applications.

### Does this support segmentation models?

**Answer:** Not currently. Segmentation support is under development for aiScope. This pipeline focuses on detection and classification models.

### Can I run RKNN models on my PC for testing?

**Answer:** No. RKNN models are specifically designed to run on Rockchip NPU hardware and cannot be executed on standard PC processors. Your PC can only perform the conversion process (PyTorch → ONNX → RKNN) using RKNN Toolkit2, but actual inference with `.rknn` files requires deployment on a physical device with a Rockchip NPU (RK3562, RK3566, RK3568, RK3576, or RK3588). For testing and validation, use the PyTorch or ONNX versions of your model on PC before converting to RKNN format for final deployment.

### How do I optimize model accuracy for medical images?

**Answer:**
1. **Fine-tune with domain-specific data** - Use surgical/medical images, not general objects
2. **Augmentation** - Medical images have unique lighting/color properties
3. **Class balance** - Ensure representative samples of each condition
4. **Validation** - Test on held-out data from different patients/procedures
5. **Calibration** - Use diverse quantization images matching deployment conditions

---

**Ready to get started?** Jump to the [Quick Start](#quick-start) section above! 🚀
