# VideoMAE CVS Classification on Endoscapes Dataset

## 🎯 Project Overview

This project fine-tunes a **VideoMAE** (Video Masked Autoencoder) model on the **Endoscapes** dataset for **CVS (Critical View of Safety) classification** during laparoscopic cholecystectomy.

### CVS Criteria
The model predicts three binary criteria:
| Criterion | Description |
|-----------|-------------|
| **C1: Two Structures** | Only cystic duct and cystic artery seen entering gallbladder |
| **C2: Hepatocystic Triangle** | Triangle of Calot cleared of fat and fibrous tissue |
| **C3: Cystic Plate** | Lower part of gallbladder separated from liver bed |

When **all three criteria** are met, CVS is considered **achieved** ✅

---

## 📁 Project Structure

```
CVS Classification/
├── config.py              # Centralized configuration (hyperparams, paths)
├── download_dataset.py    # Download & prepare Endoscapes dataset
├── dataset.py             # EndoscapesCVSDataset class (auto-detects layout)
├── dataloader.py          # DataLoader factory with weighted sampling
├── model.py               # VideoMAE classifier with custom head
├── train.py               # Training pipeline with AMP, grad norms
├── test.py                # Test evaluation with optimal thresholds
├── inference.py           # Single/batch video inference
├── evaluate.py            # Comprehensive metrics computation
├── utils.py               # Logging, checkpointing, plotting
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🏗️ Architecture

```
VideoMAE Encoder (pretrained, fine-tuned)
    │
    ▼
Mean Pooling over all tokens
    │
    ▼
LayerNorm → Dropout(0.3) → FC(768→256) → GELU → Dropout(0.15) → FC(256→3)
    │
    ▼
Sigmoid → [C1, C2, C3] probabilities
```

- **Base model**: `MCG-NJU/videomae-base` (pretrained on Kinetics-400)
- **Input**: 16 frames × 224×224, sampled at stride 4
- **Output**: 3 sigmoid probabilities (multi-label binary classification)
- **Loss**: BCEWithLogitsLoss with positive class weighting

---

## 📦 Dataset Setup

### 1. Automated Download (Recommended)

Use the included download script to automatically fetch and prepare the dataset:

```bash
# Download and prepare in the default location (./data/endoscapes)
python download_dataset.py

# Custom output directory
python download_dataset.py --output_dir "D:\datasets\endoscapes"

# If you already downloaded the zip manually
python download_dataset.py --skip_download --zip_path "D:\downloads\endoscapes.zip"

# If you already extracted the raw data
python download_dataset.py --prepare_only --extracted_dir "D:\raw_endoscapes"

# Validate an existing prepared dataset
python download_dataset.py --validate_only --output_dir "D:\datasets\endoscapes"
```

The dataset (~9-12 GB) is downloaded from the official [CAMMA/University of Strasbourg](https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip) source. It is also available on [PhysioNet](https://physionet.org/content/endoscapes-cvs201/) (credentialed access).

> **License**: CC BY-NC-SA 4.0 (non-commercial research only)

### 2. Supported Dataset Layouts

The dataset class **auto-detects** which layout is present:

**Layout A — Raw Endoscapes** (as downloaded):
```
endoscapes/
├── train/
│   ├── 1_29375.jpg                # {VIDEO_ID}_{FRAME_NUM}.jpg
│   ├── 2_29075.jpg
│   ├── annotation_ds_coco.json    # CVS annotations (COCO format)
│   └── ...
├── val/
├── test/
├── all_metadata.csv               # Consensus CVS labels from 3 annotators
├── train_vids.txt                 # 120 training videos
├── val_vids.txt                   # 41 validation videos
└── test_vids.txt                  # 40 testing videos
```

**Layout B — Prepared** (after running `download_dataset.py`):
```
endoscapes/
├── frames/
│   ├── 1/                         # Frames grouped by video ID
│   │   ├── 1_29375.jpg
│   │   ├── 1_29380.jpg
│   │   └── ...
│   └── ...
├── all_metadata.csv
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 3. CVS Annotations

- **11,090 frames** from **201 videos** annotated by **3 clinical experts**
- CVS labels are decimal values (consensus average); binarized at threshold 0.5
- Official splits: 120 train / 41 val / 40 test videos
- Labels in `all_metadata.csv` and COCO-format JSON files

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `config.py` or set environment variables:
```bash
set ENDOSCAPES_ROOT=D:\path\to\endoscapes
set OUTPUT_DIR=D:\CVS Classification\outputs
```

### 3. Train

```bash
# Train for 4000 iterations (default), checkpoint every 250 iterations
python train.py --data_root "D:\path\to\endoscapes" --max_iters 4000 --batch_size 4 --lr 1e-4

# Custom training length and checkpoint interval
python train.py --max_iters 8000 --checkpoint_interval 500 --val_interval 500
```

### 4. Test

```bash
# Evaluate using latest checkpoint (default)
python test.py --find_optimal_threshold

# Evaluate a specific checkpoint
python test.py --checkpoint outputs/checkpoints/checkpoint_iter_002000.pth
```

### 5. Inference

```bash
# Single video
python inference.py --input "D:\path\to\video_frames" --checkpoint outputs/checkpoints/best_model.pth

# Batch mode (multiple videos)
python inference.py --input "D:\path\to\all_videos" --batch_mode --sliding_window

# Sliding window for long videos
python inference.py --input "D:\path\to\video_frames" --sliding_window --window_stride 8
```

---

## 📊 Metrics Computed

### Overall Multi-Label Metrics
- Exact Match Ratio (Subset Accuracy)
- Hamming Loss
- Micro/Macro/Weighted F1 Score
- Micro/Macro AUROC
- Micro/Macro Average Precision (mAP)
- Jaccard Score (IoU)

### Per-Criterion Metrics
- Accuracy, Precision, Recall, F1
- Specificity, NPV
- AUROC, Average Precision
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Confusion Matrix

### CVS Achievement Metrics
- Accuracy, Precision, Recall, F1, AUROC for the composite "CVS achieved" label

---

## 📈 Training Outputs

After training, the following are generated in `outputs/`:

```
outputs/
├── checkpoints/
│   ├── latest_checkpoint.pth         # Always-updated latest checkpoint
│   ├── best_model.pth                # Best validation AUROC
│   ├── checkpoint_iter_000250.pth    # Every 250 iterations
│   ├── checkpoint_iter_000500.pth
│   ├── checkpoint_iter_000750.pth
│   └── ...
├── logs/
│   └── training_XXXXXXXX.log
├── plots/
│   ├── training_dashboard.png      # Loss, accuracy, LR, AUROC, overfitting monitor
│   ├── gradient_norms_detailed.png # Per-step grad norms + histogram
│   └── confusion_matrices.png     # Per-criterion confusion matrices
└── results/
    ├── training_history.json
    ├── evaluation_metrics.json
    ├── evaluation_report.txt
    └── test_predictions.csv
```

---

## ⚙️ Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `MCG-NJU/videomae-base` | Pretrained VideoMAE checkpoint |
| `NUM_FRAMES` | 16 | Frames per clip |
| `FRAME_SAMPLE_RATE` | 4 | Temporal stride |
| `IMAGE_SIZE` | 224 | Input resolution |
| `BATCH_SIZE` | 4 | Training batch size |
| `MAX_ITERS` | 4000 | Total training iterations |
| `CHECKPOINT_INTERVAL` | 250 | Save checkpoint every N iterations |
| `VAL_INTERVAL` | 250 | Validate every N iterations |
| `WARMUP_ITERS` | 500 | LR warmup iterations |
| `LEARNING_RATE` | 1e-4 | Initial learning rate |
| `MAX_GRAD_NORM` | 1.0 | Gradient clip threshold |
| `USE_AMP` | True | Mixed precision training |

---

## 🔧 Advanced Usage

### Freeze Backbone (Feature Extraction Only)
```bash
python train.py --freeze_backbone
```

### Partial Layer Freezing
```bash
python train.py --freeze_layers 8   # Freeze first 8 encoder layers
```

### Resume Training
```bash
python train.py --resume outputs/checkpoints/checkpoint_iter_002000.pth
```

### Use VideoMAE-Large
```python
# In config.py:
MODEL_NAME = "MCG-NJU/videomae-large"
```

---

## 📝 Citation

If you use this code, please cite:
```bibtex
@article{murali2023endoscapes,
  title={The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment},
  author={Murali, Aditya and others},
  journal={arXiv preprint arXiv:2312.12429},
  year={2023}
}

@inproceedings{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  booktitle={NeurIPS},
  year={2022}
}
```
