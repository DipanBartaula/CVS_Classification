"""
Configuration file for VideoMAE CVS Classification on Endoscapes Dataset.
All hyperparameters and paths are centralized here.
"""
import os
import torch

# ─────────────────────────────────────────────
#  Environment Detection
# ─────────────────────────────────────────────
# Auto-detect if running in Google Colab
IN_COLAB = os.path.exists("/content") and os.path.isdir("/content")

# ─────────────────────────────────────────────
#  Paths  (auto-configured for Colab vs Local)
# ─────────────────────────────────────────────
if IN_COLAB:
    _BASE_DIR = "/content"
    _DEFAULT_DATASET_ROOT = os.path.join(_BASE_DIR, "data", "endoscapes")
    _DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_DATASET_ROOT = os.path.join(_BASE_DIR, "data", "endoscapes")
    _DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")

DATASET_ROOT = os.environ.get("ENDOSCAPES_ROOT", _DEFAULT_DATASET_ROOT)

# Output directories
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories
for _dir in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────
#  Model Configuration
# ─────────────────────────────────────────────
# Pre-trained VideoMAE checkpoint from HuggingFace
MODEL_NAME = "MCG-NJU/videomae-base"  # Options: videomae-base, videomae-large, videomae-base-finetuned-kinetics
NUM_CLASSES = 3  # C1 (Two Structures), C2 (Hepatocystic Triangle), C3 (Cystic Plate)
CVS_CRITERIA_NAMES = [
    "C1: Two Structures",
    "C2: Hepatocystic Triangle",
    "C3: Cystic Plate",
]

# ─────────────────────────────────────────────
#  Video / Frame Configuration
# ─────────────────────────────────────────────
NUM_FRAMES = 16           # Number of frames to sample per clip (VideoMAE default)
FRAME_SAMPLE_RATE = 4     # Temporal stride for frame sampling
IMAGE_SIZE = 224          # Input resolution for VideoMAE
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet normalization

# ─────────────────────────────────────────────
#  Training Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE = 4
NUM_WORKERS = 2 if IN_COLAB else 4

# Iteration-based training
MAX_ITERS = 8000              # Total training iterations
CHECKPOINT_INTERVAL = 1000    # Save checkpoint every N iterations
VAL_INTERVAL = 1000           # Validate every N iterations
LOG_INTERVAL = 50             # Log training metrics every N iterations
WARMUP_ITERS = 500            # LR warmup iterations

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
MIN_LR = 1e-6
LABEL_SMOOTHING = 0.0
DROP_PATH_RATE = 0.1

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Mixed precision training
USE_AMP = True

# ─────────────────────────────────────────────
#  Optimizer & Scheduler
# ─────────────────────────────────────────────
OPTIMIZER = "adamw"       # Options: adamw, sgd
SCHEDULER = "cosine"      # Options: cosine, step, plateau

# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────
EVAL_INTERVAL = 1000          # Evaluate every N iterations (same as VAL_INTERVAL)
CLASSIFICATION_THRESHOLD = 0.5  # Threshold for binary predictions

# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────
SEED = 42
