"""
Training script for VideoMAE CVS Classification.

Iteration-based training pipeline with:
- Train for a fixed number of iterations (default: 4000)
- Checkpoint saved every N iterations (default: 250)
- Mixed precision training (AMP)
- Gradient clipping & gradient norm logging
- Cosine annealing with warmup
- Periodic validation
- Latest checkpoint used for final evaluation
- Training curves plotting (loss, accuracy, AUROC, grad norms)
"""
import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import config
from model import build_model
from dataloader import create_dataloaders
from evaluate import compute_metrics, print_metrics_summary
from utils import (
    set_seed,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    compute_gradient_norm,
    AverageMeter,
    TrainingHistory,
    plot_training_curves,
    plot_gradient_norms_detailed,
    format_metrics,
    count_parameters,
)

logger = logging.getLogger("CVS_Classification")


# ─────────────────────────────────────────────
#  Auto-download dataset if missing
# ─────────────────────────────────────────────
def ensure_dataset_available(data_root: str) -> str:
    """
    Check if the Endoscapes dataset is present at data_root.
    If not, automatically download and prepare it.

    Returns:
        Path to the ready-to-use dataset root.
    """
    # Check for either 'prepared' layout (frames/) or 'raw' layout (train/)
    has_prepared = os.path.isdir(os.path.join(data_root, "frames"))
    has_raw = os.path.isdir(os.path.join(data_root, "train"))
    has_metadata = os.path.exists(os.path.join(data_root, "all_metadata.csv"))

    if (has_prepared or has_raw) and has_metadata:
        logging.info(f"✅ Dataset found at: {data_root}")
        return data_root

    logging.info(f"⚠️  Dataset NOT found at: {data_root}")
    logging.info("   Will download and prepare the Endoscapes dataset automatically...")
    logging.info("   (This is a one-time operation, ~9-12 GB download)")

    try:
        from download_dataset import (
            download_file,
            extract_zip,
            find_dataset_root,
            prepare_for_videomae,
            validate_dataset,
            DATASET_URL,
            DATASET_FILENAME,
        )
    except ImportError:
        logging.error(
            "Cannot import download_dataset.py — make sure it is in the same directory.\n"
            "You can also download manually:\n"
            f"  1. Download: {DATASET_URL if 'DATASET_URL' in dir() else 'https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip'}\n"
            f"  2. Extract and point --data_root to the extracted directory."
        )
        raise FileNotFoundError(f"Dataset not found at: {data_root}")

    # Determine download location
    download_dir = os.path.join(os.path.dirname(data_root), "downloads")
    os.makedirs(download_dir, exist_ok=True)
    zip_path = os.path.join(download_dir, DATASET_FILENAME)

    # Step 1: Download
    logging.info("\n" + "━" * 50)
    logging.info("AUTO-DOWNLOAD STEP 1: Downloading dataset")
    logging.info("━" * 50)
    download_file(DATASET_URL, zip_path)

    # Step 2: Extract
    logging.info("\n" + "━" * 50)
    logging.info("AUTO-DOWNLOAD STEP 2: Extracting dataset")
    logging.info("━" * 50)
    extract_dir = os.path.join(download_dir, "endoscapes_raw")
    extract_zip(zip_path, extract_dir)

    # Find actual dataset root within extracted files
    raw_root = find_dataset_root(extract_dir)
    logging.info(f"Dataset root found at: {raw_root}")

    # Step 3: Prepare for VideoMAE
    logging.info("\n" + "━" * 50)
    logging.info("AUTO-DOWNLOAD STEP 3: Preparing for VideoMAE training")
    logging.info("━" * 50)
    prepare_for_videomae(raw_root, data_root)

    # Step 4: Validate
    validate_dataset(data_root)

    logging.info(f"\n✅ Dataset ready at: {data_root}")
    return data_root


# ─────────────────────────────────────────────
#  Iteration-based training defaults
# ─────────────────────────────────────────────
DEFAULT_MAX_ITERS = 8000
DEFAULT_CHECKPOINT_INTERVAL = 1000
DEFAULT_VAL_INTERVAL = 1000      # Validate every N iterations
DEFAULT_LOG_INTERVAL = 50        # Log metrics every N iterations
DEFAULT_WARMUP_ITERS = 500       # Warmup iterations


def parse_args():
    parser = argparse.ArgumentParser(description="Train VideoMAE for CVS Classification")
    parser.add_argument("--data_root", type=str, default=config.DATASET_ROOT,
                        help="Path to Endoscapes dataset root")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME,
                        help="HuggingFace pretrained model name")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)

    # Iteration-based training
    parser.add_argument("--max_iters", type=int, default=DEFAULT_MAX_ITERS,
                        help="Total number of training iterations (default: 8000)")
    parser.add_argument("--checkpoint_interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
                        help="Save checkpoint every N iterations (default: 1000)")
    parser.add_argument("--val_interval", type=int, default=DEFAULT_VAL_INTERVAL,
                        help="Validate every N iterations (default: 1000)")
    parser.add_argument("--log_interval", type=int, default=DEFAULT_LOG_INTERVAL,
                        help="Log training metrics every N iterations (default: 50)")
    parser.add_argument("--warmup_iters", type=int, default=DEFAULT_WARMUP_ITERS,
                        help="Number of warmup iterations (default: 500)")

    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--val_subset_size", type=int, default=250,
                        help="Number of random samples to use for periodic validation steps (0 for full set)")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the VideoMAE backbone")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Number of encoder layers to freeze")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    return parser.parse_args()


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build AdamW optimizer with parameter grouping (no weight decay on bias/norm)."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_iters: int,
    warmup_iters: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build cosine annealing scheduler with linear warmup (iteration-based)."""

    def lr_lambda(current_step):
        if current_step < warmup_iters:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_iters))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_iters) / float(
                max(1, max_iters - warmup_iters)
            )
            return max(
                config.MIN_LR / config.LEARNING_RATE,
                0.5 * (1.0 + np.cos(np.pi * progress))
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def infinite_data_loader(loader):
    """Wrap a DataLoader to cycle infinitely for iteration-based training."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    current_iter: int,
    max_iters: int,
    pos_weight: Optional[torch.Tensor] = None,
    subset_size: Optional[int] = None,
) -> Tuple[float, float, Dict]:
    """
    Validate the model.

    Returns:
        Tuple: (avg_loss, avg_accuracy, metrics_dict)
    """
    model.eval()

    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_accuracy")

    # If subset_size is specified, evaluate only on a random subset
    dataset = val_loader.dataset
    if subset_size is not None and 0 < subset_size < len(dataset):
        # Deterministic seed for validation subset per iteration for consistency
        g = torch.Generator()
        g.manual_seed(config.SEED + current_iter)
        indices = torch.randperm(len(dataset), generator=g)[:subset_size].tolist()
        
        subset_dataset = torch.utils.data.Subset(dataset, indices)
        eval_loader = torch.utils.data.DataLoader( # Changed from DataLoader to torch.utils.data.DataLoader
            subset_dataset,
            batch_size=val_loader.batch_size,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            collate_fn=val_loader.collate_fn,
            shuffle=False,
        )
        # Fix: the subset needs the same layout info if it uses it (though evaluate doesn't)
    else:
        eval_loader = val_loader

    all_labels = []
    all_probs = []
    all_preds = []

    pbar = tqdm(
        eval_loader,
        desc=f"Iter {current_iter}/{max_iters} [Val Sub]" if subset_size else f"Iter {current_iter}/{max_iters} [Val]",
        leave=True,
        dynamic_ncols=True,
    )

    for batch in pbar:
        if batch is None:
            continue

        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)

        outputs = model(pixel_values=pixel_values)
        loss = model.compute_loss(outputs["logits"], labels, pos_weight=pos_weight)

        probs = outputs["probabilities"]
        preds = (probs >= config.CLASSIFICATION_THRESHOLD).float()
        correct = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), pixel_values.size(0))
        acc_meter.update(correct, pixel_values.size(0))

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.4f}",
        })

    # Compute detailed metrics
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Compute per-criterion AUROC for tracking
    per_criterion_auroc = {}
    for i, name in enumerate(config.CVS_CRITERIA_NAMES):
        try:
            auroc = float(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        except Exception:
            auroc = 0.0
        per_criterion_auroc[name] = auroc

    val_metrics = {
        "val_loss": loss_meter.avg,
        "val_accuracy": acc_meter.avg,
        "per_criterion_auroc": per_criterion_auroc,
    }

    return loss_meter.avg, acc_meter.avg, val_metrics


def train(args):
    """Main iteration-based training function."""
    # Setup
    set_seed(args.seed)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("  VideoMAE CVS Classification — Iteration-Based Training")
    logger.info("=" * 70)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.data_root}")
    logger.info(f"Max iterations: {args.max_iters}")
    logger.info(f"Checkpoint every: {args.checkpoint_interval} iters")
    logger.info(f"Validation every: {args.val_interval} iters")
    logger.info(f"Warmup: {args.warmup_iters} iters")
    logger.info(f"Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"AMP: {not args.no_amp}")

    # Ensure dataset is available (auto-download if missing)
    args.data_root = ensure_dataset_available(args.data_root)

    # Create dataloaders
    logger.info("\n📦 Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if train_loader is None or len(train_loader.dataset) == 0:
        logger.error("❌ Training dataset is empty. Please check your data_root, splits, and metadata.")
        logger.error("   Ensure the video IDs in splits/*.txt match the 'vid' column in your metadata CSV.")
        return
    if val_loader is None or len(val_loader.dataset) == 0:
        logger.warning("⚠️  Validation dataset is empty. Validation steps will be skipped.")

    # Get class weights for imbalanced data
    pos_weight = train_loader.dataset.get_class_weights().to(config.DEVICE)
    logger.info(f"Positive class weights: {pos_weight.tolist()}")

    # Build model
    logger.info("\n🏗️  Building model...")
    model = build_model(
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        freeze_layers=args.freeze_layers,
    )

    total, trainable = count_parameters(model)
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Build optimizer & scheduler (iteration-based)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
    )

    # AMP scaler
    use_amp = not args.no_amp and config.DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume from checkpoint
    start_iter = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_iter = checkpoint.get("iteration", 0)
        logger.info(f"Resumed from iteration {start_iter}")

    # Training history & tracking
    history = TrainingHistory()
    all_step_grad_norms = []
    best_val_auroc = 0.0

    # Per-interval accumulators
    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc")
    grad_norm_meter = AverageMeter("grad_norm")

    # ──────────────────────────────────────────
    #  Iteration-Based Training Loop
    # ──────────────────────────────────────────
    logger.info(f"\n🚀 Starting training for {args.max_iters} iterations...")
    logger.info(f"   Train loader has {len(train_loader)} batches/epoch; will cycle as needed.")
    training_start_time = time.time()

    model.train()
    train_iter = infinite_data_loader(train_loader)

    pbar = tqdm(
        range(start_iter + 1, args.max_iters + 1),
        desc="Training",
        initial=start_iter,
        total=args.max_iters,
        dynamic_ncols=True,
    )

    latest_checkpoint_path = None

    for current_iter in pbar:
        # ── Get next batch ──
        batch = next(train_iter)
        if batch is None:
            continue

        pixel_values = batch["pixel_values"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward + Backward ──
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(pixel_values=pixel_values)
                loss = model.compute_loss(outputs["logits"], labels, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient norm BEFORE clipping
            grad_norm = compute_gradient_norm(model)
            all_step_grad_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values)
            loss = model.compute_loss(outputs["logits"], labels, pos_weight=pos_weight)

            loss.backward()

            # Gradient norm BEFORE clipping
            grad_norm = compute_gradient_norm(model)
            all_step_grad_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            optimizer.step()

        scheduler.step()

        # ── Compute accuracy ──
        with torch.no_grad():
            preds = (outputs["probabilities"] >= config.CLASSIFICATION_THRESHOLD).float()
            correct = (preds == labels).float().mean().item()

        # Update meters
        loss_meter.update(loss.item(), pixel_values.size(0))
        acc_meter.update(correct, pixel_values.size(0))
        grad_norm_meter.update(grad_norm)

        current_lr = scheduler.get_last_lr()[0]

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.4f}",
            "grad": f"{grad_norm:.2f}",
            "lr": f"{current_lr:.2e}",
        })

        # ── Log every log_interval iterations ──
        if current_iter % args.log_interval == 0:
            logger.info(
                f"Iter {current_iter}/{args.max_iters} | "
                f"Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.4f} | "
                f"Grad: {grad_norm_meter.avg:.4f} | LR: {current_lr:.2e}"
            )

        # ── Save checkpoint every checkpoint_interval iterations ──
        if current_iter % args.checkpoint_interval == 0:
            ckpt_filename = os.path.join(
                config.CHECKPOINT_DIR, f"checkpoint_iter_{current_iter:06d}.pth"
            )
            ckpt_data = {
                "iteration": current_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": {
                    "train_loss": loss_meter.avg,
                    "train_acc": acc_meter.avg,
                    "grad_norm": grad_norm_meter.avg,
                    "lr": current_lr,
                },
            }
            torch.save(ckpt_data, ckpt_filename)
            logger.info(f"💾 Checkpoint saved: {ckpt_filename}")

            # Also save as 'latest_checkpoint.pth' for easy resuming
            latest_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
            torch.save(ckpt_data, latest_path)
            latest_checkpoint_path = latest_path
            logger.info(f"💾 Latest checkpoint updated: {latest_path}")

        # ── Validate every val_interval iterations ──
        if current_iter % args.val_interval == 0:
            val_loss, val_acc, val_metrics = validate(
                model=model,
                val_loader=val_loader,
                current_iter=current_iter,
                max_iters=args.max_iters,
                pos_weight=pos_weight,
                subset_size=args.val_subset_size,
            )

            per_criterion_auroc = val_metrics.get("per_criterion_auroc", {})
            mean_auroc = np.mean(list(per_criterion_auroc.values())) if per_criterion_auroc else 0.0

            # Track best
            is_best = mean_auroc > best_val_auroc
            if is_best:
                best_val_auroc = mean_auroc
                # Save best model
                best_ckpt = {
                    "iteration": current_iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": val_metrics,
                }
                best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(best_ckpt, best_path)
                logger.info(f"⭐ New best model saved! AUROC: {best_val_auroc:.4f}")

            # Update history (use iteration as the x-axis "epoch")
            history.update(
                epoch=current_iter,  # Using iteration number
                train_loss=loss_meter.avg,
                val_loss=val_loss,
                train_acc=acc_meter.avg,
                val_acc=val_acc,
                lr=current_lr,
                grad_norm=grad_norm_meter.avg,
                per_criterion_auroc=per_criterion_auroc,
            )

            # Log validation summary
            logger.info(
                f"\n{'─' * 60}"
                f"\nIter {current_iter}/{args.max_iters} — Validation Summary:"
                f"\n  Train Loss: {loss_meter.avg:.4f} | Train Acc: {acc_meter.avg:.4f}"
                f"\n  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}"
                f"\n  Mean AUROC: {mean_auroc:.4f} | Best AUROC: {best_val_auroc:.4f}"
                f"\n  LR: {current_lr:.2e} | Avg Grad Norm: {grad_norm_meter.avg:.4f}"
                f"\n  Per-Criterion AUROC:"
            )
            for cname, auroc in per_criterion_auroc.items():
                logger.info(f"    {cname}: {auroc:.4f}")

            if is_best:
                logger.info(f"  ⭐ New best! (AUROC: {best_val_auroc:.4f})")

            logger.info(f"{'─' * 60}\n")

            # Reset meters after validation
            loss_meter.reset()
            acc_meter.reset()
            grad_norm_meter.reset()

            # Resume training mode
            model.train()

    total_time = time.time() - training_start_time
    logger.info(f"\n✅ Training complete! {args.max_iters} iterations in {total_time / 60:.1f} minutes")

    # ──────────────────────────────────────────
    #  Save & Plot Results
    # ──────────────────────────────────────────
    logger.info("\n📊 Generating training plots...")

    # Save training history
    history.save()

    # Plot training curves
    plot_training_curves(history, save_dir=config.PLOT_DIR)

    # Plot detailed gradient norms
    plot_gradient_norms_detailed(all_step_grad_norms, save_dir=config.PLOT_DIR)

    # ──────────────────────────────────────────
    #  Final Test Evaluation — Use LATEST checkpoint
    # ──────────────────────────────────────────
    latest_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")

    if os.path.exists(latest_ckpt_path):
        logger.info(f"\n📋 Evaluating on test set using LATEST checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        eval_iter = checkpoint.get("iteration", args.max_iters)
        logger.info(f"   Loaded checkpoint from iteration {eval_iter}")
    else:
        logger.info("\n📋 Evaluating on test set using current model (final iteration)...")

    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            if batch is None:
                continue
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            labels = batch["labels"]

            outputs = model(pixel_values=pixel_values)
            probs = outputs["probabilities"].cpu().numpy()
            preds = (probs >= config.CLASSIFICATION_THRESHOLD).astype(int)

            all_labels.append(labels.numpy())
            all_probs.append(probs)
            all_preds.append(preds)

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    test_metrics = compute_metrics(
        all_labels, all_probs, all_preds,
        save_dir=config.RESULTS_DIR,
    )

    print_metrics_summary(test_metrics)

    logger.info("\n🎉 All done! Check outputs at: " + config.OUTPUT_DIR)

    return test_metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
