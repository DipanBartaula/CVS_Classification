"""
Test script for VideoMAE CVS Classification.

Loads the latest trained checkpoint and evaluates it on the test split,
computing all relevant metrics and generating visualizations.
"""
import os
import sys
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Optional

import config
from model import build_model
from dataloader import create_dataloaders, get_single_dataset
from evaluate import compute_metrics, compute_optimal_thresholds, print_metrics_summary
from utils import (
    set_seed,
    setup_logging,
    load_checkpoint,
    plot_confusion_matrices,
)

logger = logging.getLogger("CVS_Classification")


def parse_args():
    parser = argparse.ArgumentParser(description="Test VideoMAE CVS Classifier")
    parser.add_argument("--data_root", type=str, default=config.DATASET_ROOT,
                        help="Path to Endoscapes dataset root")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth"),
                        help="Path to model checkpoint (default: latest_checkpoint.pth)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--output_dir", type=str, default=config.RESULTS_DIR,
                        help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=config.CLASSIFICATION_THRESHOLD,
                        help="Classification threshold")
    parser.add_argument("--find_optimal_threshold", action="store_true",
                        help="Find and use optimal threshold per criterion")
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


@torch.no_grad()
def run_test(
    model: torch.nn.Module,
    test_loader,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
    device: torch.device = config.DEVICE,
) -> Dict:
    """
    Run model on test set and collect predictions.
    
    Returns:
        Dict with labels, probabilities, predictions, and video_ids.
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    all_preds = []
    all_video_ids = []
    
    for batch in tqdm(test_loader, desc="🧪 Testing", dynamic_ncols=True):
        if batch is None:
            continue
        
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].numpy()
        video_ids = batch["video_ids"]
        
        outputs = model(pixel_values=pixel_values)
        probs = outputs["probabilities"].cpu().numpy()
        preds = (probs >= threshold).astype(int)
        
        all_labels.append(labels)
        all_probs.append(probs)
        all_preds.append(preds)
        all_video_ids.extend(video_ids)
    
    return {
        "labels": np.concatenate(all_labels, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "preds": np.concatenate(all_preds, axis=0),
        "video_ids": all_video_ids,
    }


def save_predictions(results: Dict, output_dir: str):
    """Save per-sample predictions to CSV."""
    import pandas as pd
    
    records = []
    for i in range(len(results["video_ids"])):
        record = {
            "video_id": results["video_ids"][i],
            "C1_true": int(results["labels"][i, 0]),
            "C2_true": int(results["labels"][i, 1]),
            "C3_true": int(results["labels"][i, 2]),
            "C1_prob": float(results["probs"][i, 0]),
            "C2_prob": float(results["probs"][i, 1]),
            "C3_prob": float(results["probs"][i, 2]),
            "C1_pred": int(results["preds"][i, 0]),
            "C2_pred": int(results["preds"][i, 1]),
            "C3_pred": int(results["preds"][i, 2]),
            "cvs_true": int(results["labels"][i].sum() == 3),
            "cvs_pred": int(results["preds"][i].sum() == 3),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved to: {csv_path}")
    
    return df


def test(args):
    """Main test function."""
    set_seed(args.seed)
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("  VideoMAE CVS Classification — Test Evaluation")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.data_root}")
    logger.info(f"Threshold: {args.threshold}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build model and load weights
    logger.info("\n🏗️  Loading model...")
    model = build_model()
    model, loaded_iter, saved_metrics = load_checkpoint(
        model, checkpoint_path=args.checkpoint
    )
    logger.info(f"Model loaded from iteration {loaded_iter}")
    
    # Create test dataloader
    logger.info("\n📦 Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Run test
    logger.info("\n🧪 Running test evaluation...")
    results = run_test(model, test_loader, threshold=args.threshold)
    
    logger.info(f"Test samples: {len(results['labels'])}")
    
    # Find optimal thresholds if requested
    if args.find_optimal_threshold:
        logger.info("\n🔍 Finding optimal thresholds per criterion...")
        optimal_thresholds = compute_optimal_thresholds(
            results["labels"], results["probs"]
        )
        logger.info("Optimal thresholds:")
        for name, thresh in optimal_thresholds.items():
            logger.info(f"  {name}: {thresh:.4f}")
        
        # Re-predict with optimal thresholds
        for i, name in enumerate(config.CVS_CRITERIA_NAMES):
            results["preds"][:, i] = (
                results["probs"][:, i] >= optimal_thresholds[name]
            ).astype(int)
        
        logger.info("Re-computed predictions with optimal thresholds")
    
    # Compute metrics
    logger.info("\n📊 Computing metrics...")
    metrics = compute_metrics(
        all_labels=results["labels"],
        all_probs=results["probs"],
        all_preds=results["preds"],
        save_dir=args.output_dir,
    )
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save predictions
    save_predictions(results, args.output_dir)
    
    logger.info(f"\n✅ Test evaluation complete! Results saved to: {args.output_dir}")
    
    return metrics


if __name__ == "__main__":
    args = parse_args()
    test(args)
