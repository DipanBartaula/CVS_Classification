"""
DataLoader factory for Endoscapes CVS Classification.

Creates train, validation, and test DataLoaders with proper configurations
including collation, sampling, and the VideoMAE image processor.
"""
import logging
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import VideoMAEImageProcessor
from typing import Dict, Optional, Tuple

import config
from dataset import EndoscapesCVSDataset

logger = logging.getLogger("CVS_Classification")


def get_processor() -> VideoMAEImageProcessor:
    """
    Load the VideoMAE image processor from HuggingFace.
    Handles resizing, normalization, and formatting for the model.
    """
    processor = VideoMAEImageProcessor.from_pretrained(
        config.MODEL_NAME,
        size={"shortest_edge": config.IMAGE_SIZE},
        crop_size={"height": config.IMAGE_SIZE, "width": config.IMAGE_SIZE},
    )
    logger.info(f"Loaded VideoMAEImageProcessor from {config.MODEL_NAME}")
    return processor


def collate_fn(batch):
    """
    Custom collate function for the CVS dataset.
    
    Handles batching of video clips and labels, filtering out any
    samples that failed to load.
    """
    # Filter out None entries (failed samples)
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    video_ids = [b["video_id"] for b in batch]
    
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "video_ids": video_ids,
    }


def create_dataloaders(
    root_dir: str = config.DATASET_ROOT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        root_dir: Path to dataset root.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        use_weighted_sampler: Whether to use weighted random sampling
                              for the training set (handles class imbalance).
        pin_memory: Pin memory for faster GPU transfer.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load the VideoMAE processor
    processor = get_processor()
    
    # Create datasets
    train_dataset = EndoscapesCVSDataset(
        root_dir=root_dir,
        split="train",
        num_frames=config.NUM_FRAMES,
        frame_sample_rate=config.FRAME_SAMPLE_RATE,
        processor=processor,
    )
    
    val_dataset = EndoscapesCVSDataset(
        root_dir=root_dir,
        split="val",
        num_frames=config.NUM_FRAMES,
        frame_sample_rate=config.FRAME_SAMPLE_RATE,
        processor=processor,
    )
    
    test_dataset = EndoscapesCVSDataset(
        root_dir=root_dir,
        split="test",
        num_frames=config.NUM_FRAMES,
        frame_sample_rate=config.FRAME_SAMPLE_RATE,
        processor=processor,
    )
    
    # Log class distributions
    for split_name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        dist = dataset.get_class_distribution()
        logger.info(f"\n{'='*50}")
        logger.info(f"Class distribution [{split_name}]:")
        for criterion, counts in dist.items():
            logger.info(
                f"  {criterion}: "
                f"pos={counts['positive']}, neg={counts['negative']}, "
                f"ratio={counts['pos_ratio']:.3f}"
            )
    
    # Setup weighted sampler for training (handles class imbalance)
    train_sampler = None
    train_shuffle = True
    
    if use_weighted_sampler and len(train_dataset) > 0:
        sample_weights = _compute_sample_weights(train_dataset)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False  # Sampler handles shuffling
        logger.info("Using weighted random sampler for training")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    
    logger.info(
        f"\nDataLoaders created:"
        f"\n  Train: {len(train_loader)} batches ({len(train_dataset)} samples)"
        f"\n  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)"
        f"\n  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)"
        f"\n  Batch size: {batch_size}, Workers: {num_workers}"
    )
    
    return train_loader, val_loader, test_loader


def _compute_sample_weights(dataset: EndoscapesCVSDataset) -> torch.Tensor:
    """
    Compute per-sample weights for WeightedRandomSampler.
    
    Assigns higher weight to samples with rarer label combinations
    to address class imbalance in the training set.
    """
    import numpy as np
    
    all_labels = np.array([s["labels"] for s in dataset.samples])
    
    # Compute weight per criterion based on inverse frequency
    criterion_weights = []
    for i in range(config.NUM_CLASSES):
        pos_ratio = all_labels[:, i].mean()
        pos_ratio = max(pos_ratio, 0.01)  # Avoid division by zero
        neg_ratio = 1 - pos_ratio
        criterion_weights.append({
            "pos_weight": 1.0 / (2.0 * pos_ratio),
            "neg_weight": 1.0 / (2.0 * neg_ratio),
        })
    
    # Compute per-sample weight as average across criteria
    sample_weights = np.zeros(len(all_labels))
    for i in range(len(all_labels)):
        w = 0.0
        for j in range(config.NUM_CLASSES):
            if all_labels[i, j] == 1:
                w += criterion_weights[j]["pos_weight"]
            else:
                w += criterion_weights[j]["neg_weight"]
        sample_weights[i] = w / config.NUM_CLASSES
    
    return torch.tensor(sample_weights, dtype=torch.float64)


def get_single_dataset(
    split: str = "test",
    root_dir: str = config.DATASET_ROOT,
) -> EndoscapesCVSDataset:
    """Get a single dataset split (useful for evaluation/inference)."""
    processor = get_processor()
    return EndoscapesCVSDataset(
        root_dir=root_dir,
        split=split,
        num_frames=config.NUM_FRAMES,
        frame_sample_rate=config.FRAME_SAMPLE_RATE,
        processor=processor,
    )
