# 🌿 Plant Disease Foundation Model

A Self-Supervised Learning (SSL) foundation model for plant disease classification, built with SimCLR and a ResNet-18 backbone. The model is pre-trained on unlabelled plant images and then fine-tuned for 38-class plant disease classification. GradCAM explainability is integrated for inference visualisation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [1. SSL Pre-training](#1-ssl-pre-training)
  - [2. Fine-tuning](#2-fine-tuning)
  - [3. GradCAM Inference](#3-gradcam-inference)
- [Model Architecture](#model-architecture)
- [Fine-tuning Strategies](#fine-tuning-strategies)
- [Metrics](#metrics)
- [Output Files](#output-files)
- [CI/CD](#cicd)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a two-stage transfer learning pipeline:

1. **SSL Pre-training (SimCLR)** — trains an encoder on *unlabelled* plant images using contrastive learning. No labels required.
2. **Fine-tuning** — loads the pre-trained encoder and trains a classification head on *labelled* plant disease images (38 classes).
3. **GradCAM Inference** — generates visual explanations highlighting which leaf regions influenced each prediction.

### Why Self-Supervised Learning?

| Property | SSL Approach |
|---|---|
| Labels needed for pre-training | None |
| Transfer to new tasks | Add a small classification head |
| Data efficiency at fine-tune | High — few labels needed |
| Explainability | GradCAM heatmaps per prediction |

---

## Project Structure

```
plant-disease-ssl/
│
├── model.py                  # Encoder (ResNet-18) + ProjectionHead + SimCLRModel
├── dataset.py                # SSL contrastive dataset + augmentation pipeline
├── loss.py                   # NT-Xent contrastive loss
├── train.py                  # SSL pre-training loop + checkpoint saving
├── finetune.py               # Fine-tuning classifier + full metrics reporting
├── gradcam_inference.py      # GradCAM inference + visualisation
│
├── requirements.txt          # Python dependencies
├── .gitignore
├── .gitattributes
│
├── data/                     # (not committed — see .gitignore)
│   ├── train/
│   │   ├── Apple___Apple_scab/
│   │   └── ...
│   └── val/
│       ├── Apple___Apple_scab/
│       └── ...
│
├── checkpoints/              # SSL pre-training checkpoints (not committed)
│   ├── foundation_encoder.pt
│   └── checkpoint_epoch_*.pt
│
├── finetune_checkpoints/     # Fine-tuning outputs (not committed)
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── metrics_log.csv
│   └── confusion_matrix_*.png
│
└── xai/                      # GradCAM outputs (not committed)
    ├── <ClassName>/
    │   ├── original.jpg
    │   ├── gradcam_heatmap.png
    │   ├── gradcam_overlay.png
    │   └── gradcam_panel.png
    ├── summary_grid.png
    └── confidence_report.png
```

---

## Pipeline

```
Unlabelled plant images
        │
        ▼
  ┌─────────────┐
  │ SSL Pre-    │  train.py        → checkpoints/foundation_encoder.pt
  │ training    │  (SimCLR)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ Fine-tuning │  finetune.py     → finetune_checkpoints/best_model.pt
  │ Classifier  │  (38 classes)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  GradCAM   │  gradcam_        → xai/
  │  Inference  │  inference.py
  └─────────────┘
```

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU recommended (CPU works but is slow for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/plant-disease-ssl.git
cd plant-disease-ssl

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Data Format

```
data/
  train/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___Cedar_apple_rust/
    Apple___healthy/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___healthy/
    ...  (38 classes total)
  val/
    Apple___Apple_scab/
    ...
```

> **Note:** The data directory is excluded from version control via `.gitignore`.  
> The [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) is a common source for this class structure.

---

## Usage

### 1. SSL Pre-training

Train the foundation model on unlabelled images (labels/folder names are ignored):

```bash
python train.py \
  --data_dir data \
  --epochs 100 \
  --batch_size 64 \
  --image_size 224 \
  --embed_dim 256 \
  --temperature 0.5 \
  --save_dir checkpoints
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Root folder with plant images |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `64` | Larger = more negatives = better SSL |
| `--embed_dim` | `256` | Encoder output dimension |
| `--temperature` | `0.5` | NT-Xent loss temperature |
| `--resume` | `None` | Path to checkpoint to resume from |

**Outputs:**
```
checkpoints/
  foundation_encoder.pt         ← use this for fine-tuning
  checkpoint_epoch_010.pt
  checkpoint_epoch_020.pt
  ...
```

---

### 2. Fine-tuning

Fine-tune the foundation encoder on labelled plant disease images:

```bash
python finetune.py \
  --data_dir data \
  --checkpoint_path checkpoints/foundation_encoder.pt \
  --strategy partial \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_path` | `checkpoints/foundation_encoder.pt` | Pre-trained encoder |
| `--strategy` | `partial` | `frozen` / `partial` / `full` |
| `--epochs` | `30` | Fine-tuning epochs |
| `--num_classes` | `38` | Number of disease classes |
| `--lr` | `1e-3` | Learning rate |

---

### 3. GradCAM Inference

Run inference on one image per class and generate GradCAM visualisations:

```bash
python gradcam_inference.py \
  --model_path finetune_checkpoints/best_model.pt \
  --val_dir data/val \
  --xai_dir xai \
  --seed 42
```

---

## Model Architecture

### SSL Pre-training

```
Input (B, 3, 224, 224)
        │
   ┌────┴────┐
   │ Encoder │   ResNet-18 backbone (layer1→layer4→AvgPool)
   │         │   + Linear(512 → 256) + BatchNorm1d
   └────┬────┘
        │  (B, 256)  ← saved as foundation_encoder.pt
   ┌────┴────────┐
   │ Projection  │   Linear(256→512) → BN → ReLU
   │    Head     │   Linear(512→512) → BN → ReLU
   │  [discarded]│   Linear(512→128)
   └────┬────────┘
        │  (B, 128) L2-normalised
        ▼
   NT-Xent Loss
```

### Fine-tuning

```
Input (B, 3, 224, 224)
        │
   ┌────┴────┐
   │ Encoder │   Loaded from foundation_encoder.pt
   └────┬────┘
        │  (B, 256)
   ┌────┴──────────┐
   │ Classifier    │   Linear(256→512) → BN → ReLU → Dropout(0.3)
   │    Head       │   Linear(512→256) → BN → ReLU → Dropout(0.15)
   └────┬──────────┘   Linear(256→38)
        │
   CrossEntropyLoss (label_smoothing=0.1)
```

---

## Fine-tuning Strategies

| Strategy | Trainable Params | When to Use |
|---|---|---|
| `frozen` | Head only | Few labels, fast training |
| `partial` | ResNet `layer4` + embedding fc + head | **Default** — best balance |
| `full` | Entire encoder + head | Lots of labelled data |

---

## Metrics

All metrics are computed and logged every epoch for both train and val splits:

| Metric | Description |
|---|---|
| **Loss** | Cross-entropy with label smoothing |
| **Accuracy** | Overall top-1 accuracy |
| **Top-5 Accuracy** | True label in top-5 predictions |
| **Precision (macro)** | Avg precision, all classes equal weight |
| **Precision (weighted)** | Avg precision weighted by class size |
| **Recall (macro / weighted)** | Sensitivity per class |
| **F1 (macro / weighted)** | Harmonic mean of precision + recall |
| **Per-class Accuracy** | Individual accuracy per disease class |
| **Confusion Matrix** | Saved as PNG every 5 epochs |

---

## Output Files

```
checkpoints/                        # SSL pre-training
  foundation_encoder.pt             ← encoder weights for fine-tuning

finetune_checkpoints/               # Fine-tuning
  best_model.pt                     ← best validation accuracy
  final_model.pt                    ← last epoch
  metrics_log.csv                   ← all metrics, all epochs
  confusion_matrix_epoch_030.png    ← normalised heatmap

xai/                                # GradCAM
  <ClassName>/
    original.jpg
    gradcam_heatmap.png
    gradcam_overlay.png
    gradcam_panel.png               ← Original | Heatmap | Overlay
  summary_grid.png                  ← all 38 classes in one image
  confidence_report.png             ← per-class confidence bar chart
```

---

## CI/CD

This project uses GitHub Actions for automated testing and linting.

### Workflows

| Workflow | Trigger | Description |
|---|---|---|
| `ci.yml` | Push / PR to `main`, `develop` | Lint + unit tests |
| `codeql.yml` | Push / PR + weekly schedule | Security analysis |

### Running Tests Locally

```bash
# Lint
flake8 . --max-line-length=120

# Tests
pytest tests/ -v
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
