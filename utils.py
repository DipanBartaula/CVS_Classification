"""
Utility functions for VideoMAE CVS Classification.
Includes: seed setting, logging, checkpoint management, plotting, gradient norms.
"""
import os
import json
import random
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config


# ─────────────────────────────────────────────
#  Seed & Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = config.SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed}")


# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging to console and optional file."""
    logger = logging.getLogger("CVS_Classification")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.LOG_DIR, f"training_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# ─────────────────────────────────────────────
#  Checkpoint Management
# ─────────────────────────────────────────────
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    is_best: bool = False,
    filename: Optional[str] = None,
):
    """Save model checkpoint."""
    if filename is None:
        filename = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:03d}.pth")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")
    
    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        torch.save(checkpoint, best_path)
        logging.info(f"Best model saved: {best_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, int, Dict]:
    """Load model checkpoint."""
    if checkpoint_path is None:
        # Default to latest checkpoint (iteration-based training)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            # Fall back to best model
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Support both epoch-based and iteration-based checkpoints
    step = checkpoint.get("iteration", checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})
    
    logging.info(f"Checkpoint loaded (step {step}): {checkpoint_path}")
    return model, step, metrics


# ─────────────────────────────────────────────
#  Gradient Norm Computation
# ─────────────────────────────────────────────
def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the total gradient L2 norm across all model parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_per_layer_gradient_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient norms per named parameter group."""
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = p.grad.data.norm(2).item()
    return norms


# ─────────────────────────────────────────────
#  AverageMeter
# ─────────────────────────────────────────────
class AverageMeter:
    """Track running average and current value of a metric."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ─────────────────────────────────────────────
#  Training History
# ─────────────────────────────────────────────
class TrainingHistory:
    """Record and manage training history for plotting."""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        self.gradient_norms: List[float] = []
        self.per_criterion_auroc: Dict[str, List[float]] = {
            name: [] for name in config.CVS_CRITERIA_NAMES
        }
        self.epochs: List[int] = []
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        grad_norm: float,
        per_criterion_auroc: Optional[Dict[str, float]] = None,
    ):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.gradient_norms.append(grad_norm)
        
        if per_criterion_auroc:
            for name, auroc in per_criterion_auroc.items():
                if name in self.per_criterion_auroc:
                    self.per_criterion_auroc[name].append(auroc)
    
    def save(self, filepath: Optional[str] = None):
        """Save training history to JSON."""
        if filepath is None:
            filepath = os.path.join(config.RESULTS_DIR, "training_history.json")
        
        data = {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "gradient_norms": self.gradient_norms,
            "per_criterion_auroc": self.per_criterion_auroc,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Training history saved: {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """Load training history from JSON."""
        if filepath is None:
            filepath = os.path.join(config.RESULTS_DIR, "training_history.json")
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.epochs = data["epochs"]
        self.train_losses = data["train_losses"]
        self.val_losses = data["val_losses"]
        self.train_accuracies = data["train_accuracies"]
        self.val_accuracies = data["val_accuracies"]
        self.learning_rates = data["learning_rates"]
        self.gradient_norms = data["gradient_norms"]
        self.per_criterion_auroc = data.get("per_criterion_auroc", {})


# ─────────────────────────────────────────────
#  Plotting Functions
# ─────────────────────────────────────────────
def _apply_plot_style():
    """Apply consistent styling to all plots."""
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560",
        "axes.labelcolor": "#eaeaea",
        "xtick.color": "#eaeaea",
        "ytick.color": "#eaeaea",
        "text.color": "#eaeaea",
        "grid.color": "#2a2a4a",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_training_curves(history: TrainingHistory, save_dir: Optional[str] = None):
    """Generate comprehensive training plots."""
    if save_dir is None:
        save_dir = config.PLOT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    _apply_plot_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle("VideoMAE CVS Classification — Training Dashboard", fontsize=18, fontweight="bold", color="#e94560")
    
    epochs = history.epochs
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history.train_losses, "-o", label="Train Loss", color="#00d2ff", markersize=3, linewidth=2)
    ax.plot(epochs, history.val_losses, "-s", label="Val Loss", color="#e94560", markersize=3, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(facecolor="#1a1a2e", edgecolor="#e94560")
    
    # 2. Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history.train_accuracies, "-o", label="Train Acc", color="#00d2ff", markersize=3, linewidth=2)
    ax.plot(epochs, history.val_accuracies, "-s", label="Val Acc", color="#e94560", markersize=3, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend(facecolor="#1a1a2e", edgecolor="#e94560")
    
    # 3. Learning rate
    ax = axes[0, 2]
    ax.plot(epochs, history.learning_rates, "-o", color="#7b2ff7", markersize=3, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    
    # 4. Gradient norms
    ax = axes[1, 0]
    ax.plot(epochs, history.gradient_norms, "-o", color="#ff6b6b", markersize=3, linewidth=2)
    ax.axhline(y=config.MAX_GRAD_NORM, color="#ffd93d", linestyle="--", linewidth=1.5, label=f"Clip threshold ({config.MAX_GRAD_NORM})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Norms")
    ax.legend(facecolor="#1a1a2e", edgecolor="#e94560")
    
    # 5. Per-criterion AUROC
    ax = axes[1, 1]
    colors = ["#00d2ff", "#e94560", "#7b2ff7"]
    for i, (name, auroc_values) in enumerate(history.per_criterion_auroc.items()):
        if auroc_values:
            plot_epochs = epochs[:len(auroc_values)]
            ax.plot(plot_epochs, auroc_values, "-o", label=name, color=colors[i % len(colors)], markersize=3, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Criterion AUROC")
    ax.legend(facecolor="#1a1a2e", edgecolor="#e94560", fontsize=9)
    ax.set_ylim(0, 1.05)
    
    # 6. Train vs Val loss gap (overfitting monitor)
    ax = axes[1, 2]
    if len(history.train_losses) == len(history.val_losses):
        gap = [v - t for t, v in zip(history.train_losses, history.val_losses)]
        ax.fill_between(epochs, 0, gap, alpha=0.3, color="#e94560")
        ax.plot(epochs, gap, "-o", color="#e94560", markersize=3, linewidth=2)
    ax.axhline(y=0, color="#ffd93d", linestyle="--", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Val Loss - Train Loss")
    ax.set_title("Overfitting Monitor")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, "training_dashboard.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info(f"Training dashboard saved: {save_path}")


def plot_gradient_norms_detailed(
    per_step_grad_norms: List[float],
    save_dir: Optional[str] = None,
):
    """Plot per-step gradient norms (recorded at each training step)."""
    if save_dir is None:
        save_dir = config.PLOT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    _apply_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Gradient Norm Analysis", fontsize=16, fontweight="bold", color="#e94560")
    
    steps = list(range(1, len(per_step_grad_norms) + 1))
    
    # Per-step gradient norms
    ax = axes[0]
    ax.plot(steps, per_step_grad_norms, linewidth=0.8, color="#00d2ff", alpha=0.7)
    # Moving average
    window = min(50, len(per_step_grad_norms) // 5) if len(per_step_grad_norms) > 10 else 1
    if window > 1:
        moving_avg = np.convolve(per_step_grad_norms, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(per_step_grad_norms) + 1), moving_avg, linewidth=2, color="#e94560", label=f"Moving Avg (w={window})")
    ax.axhline(y=config.MAX_GRAD_NORM, color="#ffd93d", linestyle="--", linewidth=1.5, label=f"Clip={config.MAX_GRAD_NORM}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Per-Step Gradient Norms")
    ax.legend(facecolor="#1a1a2e", edgecolor="#e94560")
    
    # Histogram
    ax = axes[1]
    ax.hist(per_step_grad_norms, bins=50, color="#7b2ff7", alpha=0.8, edgecolor="#1a1a2e")
    ax.axvline(x=config.MAX_GRAD_NORM, color="#ffd93d", linestyle="--", linewidth=2, label=f"Clip={config.MAX_GRAD_NORM}")
    ax.axvline(x=np.mean(per_step_grad_norms), color="#e94560", linestyle="-", linewidth=2, label=f"Mean={np.mean(per_step_grad_norms):.3f}")
    ax.set_xlabel("Gradient Norm")
    ax.set_ylabel("Frequency")
    ax.set_title("Gradient Norm Distribution")
    ax.legend(facecolor="#1a1a2e", edgecolor="#e94560")
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(save_dir, "gradient_norms_detailed.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info(f"Gradient norm plot saved: {save_path}")


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    save_dir: Optional[str] = None,
):
    """Plot confusion matrix for each CVS criterion."""
    if save_dir is None:
        save_dir = config.PLOT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    _apply_plot_style()
    
    n = len(confusion_matrices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    fig.suptitle("Confusion Matrices per CVS Criterion", fontsize=16, fontweight="bold", color="#e94560")
    
    if n == 1:
        axes = [axes]
    
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="RdPu",
            xticklabels=["Not Achieved", "Achieved"],
            yticklabels=["Not Achieved", "Achieved"],
            ax=ax, cbar_kws={"shrink": 0.8},
            linewidths=1, linecolor="#1a1a2e",
        )
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(save_dir, "confusion_matrices.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info(f"Confusion matrices saved: {save_path}")


def format_metrics(metrics: Dict) -> str:
    """Format metrics dictionary into a readable string."""
    lines = []
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        elif isinstance(val, dict):
            lines.append(f"  {key}:")
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, float):
                    lines.append(f"    {sub_key}: {sub_val:.4f}")
                else:
                    lines.append(f"    {sub_key}: {sub_val}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
