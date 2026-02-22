"""
Endoscapes CVS Classification Dataset.

Loads video clips from the Endoscapes dataset and provides multi-label
CVS (Critical View of Safety) annotations for VideoMAE fine-tuning.

Supports TWO dataset layouts:

Layout A — Prepared (by download_dataset.py):
    DATASET_ROOT/
    ├── frames/
    │   ├── 1/                   # Video ID subdirectory
    │   │   ├── 1_29375.jpg
    │   │   ├── 1_29380.jpg
    │   │   └── ...
    │   └── ...
    ├── all_metadata.csv
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt

Layout B — Raw Endoscapes (flat structure):
    DATASET_ROOT/
    ├── train/
    │   ├── 1_29375.jpg          # {VIDEO_ID}_{FRAME_NUM}.jpg
    │   ├── 2_29075.jpg
    │   ├── annotation_ds_coco.json
    │   └── ...
    ├── val/
    ├── test/
    ├── all_metadata.csv
    ├── train_vids.txt
    ├── val_vids.txt
    └── test_vids.txt
"""
import os
import glob
import json
import logging
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger("CVS_Classification")


class EndoscapesCVSDataset(Dataset):
    """
    Endoscapes dataset for CVS classification with VideoMAE.
    
    Each sample is a video clip (sequence of frames) centered around an
    annotated frame, with multi-label binary CVS annotations (C1, C2, C3).
    
    The dataset samples `num_frames` frames using a temporal stride of
    `frame_sample_rate` around each annotated keyframe. If there are not
    enough frames, boundary frames are repeated (temporal padding).
    
    Automatically detects dataset layout (prepared vs raw) and loads
    accordingly.
    
    Args:
        root_dir: Path to the dataset root directory.
        split: One of 'train', 'val', 'test'.
        num_frames: Number of frames to sample per clip.
        frame_sample_rate: Temporal stride between sampled frames.
        transform: Optional transform applied to each frame.
        processor: Optional HuggingFace VideoMAEImageProcessor.
    """
    
    def __init__(
        self,
        root_dir: str = config.DATASET_ROOT,
        split: str = "train",
        num_frames: int = config.NUM_FRAMES,
        frame_sample_rate: int = config.FRAME_SAMPLE_RATE,
        transform=None,
        processor=None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.transform = transform
        self.processor = processor
        
        # Auto-detect dataset layout
        self.layout = self._detect_layout()
        logger.info(f"Detected dataset layout: {self.layout}")
        
        # Paths (depend on layout)
        if self.layout == "prepared":
            self.frames_dir = os.path.join(root_dir, "frames")
            self.metadata_path = os.path.join(root_dir, "all_metadata.csv")
            self.split_file = os.path.join(root_dir, "splits", f"{split}.txt")
        else:  # "raw"
            self.frames_dir = root_dir  # frames live in split subdirs
            self.metadata_path = os.path.join(root_dir, "all_metadata.csv")
            self.split_file = os.path.join(root_dir, f"{split}_vids.txt")
        
        # Load annotations and build sample list
        self.metadata = self._load_metadata()
        self.video_ids = self._load_split_ids()
        
        # Build per-video frame index for raw layout
        if self.layout == "raw":
            self._video_frames_cache = self._index_raw_frames()
        
        self.samples = self._build_samples()
        
        logger.info(
            f"EndoscapesCVSDataset [{split}]: "
            f"{len(self.samples)} clips from {len(self.video_ids)} videos"
        )
    
    def _detect_layout(self) -> str:
        """Detect whether dataset is in 'prepared' or 'raw' layout."""
        # Prepared layout has frames/ subdirectory with video subdirectories
        frames_dir = os.path.join(self.root_dir, "frames")
        if os.path.isdir(frames_dir):
            subdirs = [d for d in os.listdir(frames_dir)
                       if os.path.isdir(os.path.join(frames_dir, d))]
            if subdirs:
                return "prepared"
        
        # Raw layout has train/, val/, test/ directories with flat frame files
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(self.root_dir, split_name)
            if os.path.isdir(split_dir):
                return "raw"
        
        # Default to prepared (will show useful errors later)
        return "prepared"
    
    def _index_raw_frames(self) -> Dict[str, List[str]]:
        """
        Build an index of frame paths per video for the raw flat layout.
        Scans the split directory for {VIDEO_ID}_{FRAME_NUM}.jpg files.
        """
        video_frames = defaultdict(list)
        
        # In raw layout, frames are in the split subdirectory
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_dir):
            # Try 'all' directory (contains all frames at 1fps)
            all_dir = os.path.join(self.root_dir, "all")
            if os.path.isdir(all_dir):
                split_dir = all_dir
            else:
                logger.warning(f"Split directory not found: {split_dir}")
                return video_frames
        
        for filename in os.listdir(split_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            basename = os.path.splitext(filename)[0]
            parts = basename.rsplit("_", 1)
            if len(parts) == 2:
                video_id = parts[0]
                frame_path = os.path.join(split_dir, filename)
                video_frames[video_id].append(frame_path)
        
        # Sort frames within each video by frame number
        for vid in video_frames:
            video_frames[vid] = sorted(
                video_frames[vid],
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[1])
            )
        
        logger.info(
            f"Indexed {sum(len(v) for v in video_frames.values())} frames "
            f"from {len(video_frames)} videos in {split_dir}"
        )
        return dict(video_frames)
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and preprocess the metadata CSV."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}\n"
                f"Expected 'all_metadata.csv' in {self.root_dir}"
            )
        
        df = pd.read_csv(self.metadata_path)
        
        # Normalize column names (handle various naming conventions)
        col_map = {}
        for col in df.columns:
            lower = col.strip().lower()
            # Video ID mapping
            if lower in ("vid", "video_id", "videoid", "video"):
                col_map[col] = "video_id"
            elif "video" in lower and "id" in lower:
                col_map[col] = "video_id"
            
            # Frame ID mapping
            elif lower in ("frame", "frame_id", "frameid", "frame_num"):
                col_map[col] = "frame_id"
            elif "frame" in lower and ("id" in lower or "num" in lower or "number" in lower):
                col_map[col] = "frame_id"
            
            # Criteria mapping
            elif lower in ("c1", "criterion_1", "two_structures"):
                col_map[col] = "C1"
            elif lower in ("c2", "criterion_2", "hepatocystic_triangle"):
                col_map[col] = "C2"
            elif lower in ("c3", "criterion_3", "cystic_plate"):
                col_map[col] = "C3"
        
        df = df.rename(columns=col_map)
        
        # Ensure required columns exist
        required = ["video_id", "frame_id", "C1", "C2", "C3"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in metadata: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Clean IDs: Convert to string, handle potential float '1.0' -> '1', and strip
        for col in ["video_id", "frame_id"]:
            # If it's float, convert to int first to remove .0
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].fillna(0).astype(int)
            df[col] = df[col].astype(str).str.strip()
        
        # Binarize CVS labels: threshold at 0.5 (handles decimal consensus values)
        for col in ["C1", "C2", "C3"]:
            # Ensure numeric
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = (df[col] >= 0.5).astype(int)
        
        return df
    
    def _load_split_ids(self) -> List[str]:
        """Load video IDs for the current split."""
        if os.path.exists(self.split_file):
            with open(self.split_file, "r") as f:
                raw_ids = [line.strip() for line in f if line.strip()]
            
            # Normalize IDs: handle float strings like '1.0' -> '1' and strip
            normalized_ids = []
            for rid in raw_ids:
                try:
                    # If it's a numeric string that looks like 1.0, convert to 1
                    if "." in rid:
                        rid = str(int(float(rid)))
                except (ValueError, TypeError):
                    pass
                normalized_ids.append(rid.strip())
            
            logger.info(f"Loaded {len(normalized_ids)} video IDs from {self.split_file}")
            return normalized_ids
        else:
            # Auto-generate splits if split files don't exist
            logger.warning(
                f"Split file not found: {self.split_file}. "
                f"Auto-generating splits from metadata..."
            )
            return self._auto_generate_splits()
    
    def _auto_generate_splits(self) -> List[str]:
        """Auto-generate train/val/test splits if split files don't exist."""
        all_video_ids = sorted(self.metadata["video_id"].unique().tolist())
        n = len(all_video_ids)
        
        np.random.seed(config.SEED)
        np.random.shuffle(all_video_ids)
        
        # 70% train, 15% val, 15% test
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        splits = {
            "train": all_video_ids[:train_end],
            "val": all_video_ids[train_end:val_end],
            "test": all_video_ids[val_end:],
        }
        
        # Save generated splits
        splits_dir = os.path.join(self.root_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        for split_name, ids in splits.items():
            split_path = os.path.join(splits_dir, f"{split_name}.txt")
            with open(split_path, "w") as f:
                f.write("\n".join(ids))
            logger.info(f"Generated {split_name} split: {len(ids)} videos -> {split_path}")
        
        return splits[self.split]
    
    def _build_samples(self) -> List[Dict]:
        """
        Build list of samples (video clips) for this split.
        
        Each sample is a dict with:
            - video_id: str
            - keyframe_idx: int (index of annotated frame in video)
            - frame_dir: str (path to video frames directory)
            - all_frames: List[str] (sorted list of all frame paths)
            - labels: np.ndarray of shape (3,) with binary CVS labels
        
        Handles both 'prepared' layout (per-video subdirectories) and
        'raw' layout (flat frame files in split directories).
        """
        samples = []
        
        # Filter metadata for this split's video IDs
        split_metadata = self.metadata[self.metadata["video_id"].isin(self.video_ids)]
        
        for _, row in split_metadata.iterrows():
            video_id = str(row["video_id"])
            frame_id = row["frame_id"]
            labels = np.array([row["C1"], row["C2"], row["C3"]], dtype=np.float32)
            
            if self.layout == "raw":
                # Raw layout: get frames from the pre-built cache
                all_frames = self._video_frames_cache.get(video_id, [])
                if not all_frames:
                    continue
                frame_dir = os.path.dirname(all_frames[0]) if all_frames else ""
            else:
                # Prepared layout: find per-video subdirectory
                frame_dir = os.path.join(self.frames_dir, video_id)
                if not os.path.isdir(frame_dir):
                    # Try alternative naming patterns
                    alt_patterns = [
                        os.path.join(self.frames_dir, f"video_{video_id}"),
                        os.path.join(self.frames_dir, f"Video_{video_id}"),
                        os.path.join(self.frames_dir, video_id.zfill(3)),
                    ]
                    frame_dir = None
                    for alt in alt_patterns:
                        if os.path.isdir(alt):
                            frame_dir = alt
                            break
                    
                    if frame_dir is None:
                        continue  # Skip videos without frame directories
                
                # Get sorted frame paths from the directory
                frame_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                all_frames = []
                for ext in frame_exts:
                    all_frames.extend(glob.glob(os.path.join(frame_dir, ext)))
                all_frames = sorted(all_frames)
            
            if len(all_frames) < 1:
                continue
            
            # Find the keyframe index
            keyframe_idx = self._find_keyframe_index(all_frames, frame_id)
            
            samples.append({
                "video_id": video_id,
                "keyframe_idx": keyframe_idx,
                "frame_dir": frame_dir,
                "all_frames": all_frames,
                "labels": labels,
            })
        
        if len(samples) == 0:
            logger.warning(
                f"No valid samples found for split '{self.split}'. "
                f"Check dataset directory structure and metadata."
            )
        
        return samples
    
    def _find_keyframe_index(self, all_frames: List[str], frame_id) -> int:
        """Find the index of a keyframe in the sorted frame list."""
        frame_id_str = str(frame_id)
        
        for i, fp in enumerate(all_frames):
            basename = os.path.splitext(os.path.basename(fp))[0]
            # Match by frame number or full frame ID
            if frame_id_str in basename:
                return i
        
        # If not found, try numeric matching
        try:
            frame_num = int(frame_id_str)
            # Return proportional index based on frame number
            if len(all_frames) > 0:
                return min(frame_num, len(all_frames) - 1)
        except ValueError:
            pass
        
        # Default: return middle frame
        return len(all_frames) // 2
    
    def _sample_frame_indices(self, total_frames: int, keyframe_idx: int) -> List[int]:
        """
        Sample frame indices for a clip centered around the keyframe.
        
        Uses temporal stride (frame_sample_rate) and handles boundary padding
        by clamping to valid frame indices.
        """
        clip_len = self.num_frames * self.frame_sample_rate
        half_clip = clip_len // 2
        
        # Center the clip around the keyframe
        start = keyframe_idx - half_clip
        # end = keyframe_idx + half_clip
        
        indices = []
        for i in range(self.num_frames):
            idx = start + i * self.frame_sample_rate
            # Clamp to valid range
            idx = max(0, min(idx, total_frames - 1))
            indices.append(idx)
        
        return indices
    
    def _load_frames(self, frame_paths: List[str], indices: List[int]) -> List[Image.Image]:
        """Load frames at specified indices as PIL Images."""
        frames = []
        for idx in indices:
            img = Image.open(frame_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return frames
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video clip and its CVS labels.
        
        Returns:
            dict with:
                - pixel_values: Tensor of shape (num_frames, C, H, W)
                - labels: Tensor of shape (3,) with binary CVS labels
                - video_id: str
        """
        sample = self.samples[idx]
        
        # Sample frame indices
        indices = self._sample_frame_indices(
            total_frames=len(sample["all_frames"]),
            keyframe_idx=sample["keyframe_idx"],
        )
        
        # Load frames
        frames = self._load_frames(sample["all_frames"], indices)
        
        # Process frames using VideoMAE processor
        if self.processor is not None:
            # Processor expects list of lists of PIL Images or numpy arrays
            inputs = self.processor(
                list(frames),
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].squeeze(0)  # (num_frames, C, H, W)
        else:
            # Manual processing: resize and normalize
            import torchvision.transforms as T
            manual_transform = T.Compose([
                T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
            ])
            pixel_values = torch.stack([manual_transform(f) for f in frames])  # (num_frames, C, H, W)
        
        labels = torch.tensor(sample["labels"], dtype=torch.float32)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "video_id": sample["video_id"],
        }
    
    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get class distribution for each CVS criterion."""
        distribution = {}
        if len(self.samples) == 0:
            for name in config.CVS_CRITERIA_NAMES:
                distribution[name] = {"positive": 0, "negative": 0, "total": 0, "pos_ratio": 0.0}
            return distribution

        all_labels = np.array([s["labels"] for s in self.samples])
        
        for i, name in enumerate(config.CVS_CRITERIA_NAMES):
            positive = int(all_labels[:, i].sum())
            negative = len(all_labels) - positive
            distribution[name] = {
                "positive": positive,
                "negative": negative,
                "total": len(all_labels),
                "pos_ratio": positive / max(len(all_labels), 1),
            }
        
        return distribution
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data using inverse frequency.
        Returns tensor of shape (3,) with weight for positive class per criterion.
        """
        if len(self.samples) == 0:
            return torch.ones(config.NUM_CLASSES)

        all_labels = np.array([s["labels"] for s in self.samples])
        weights = []
        
        for i in range(config.NUM_CLASSES):
            n_pos = all_labels[:, i].sum()
            n_neg = len(all_labels) - n_pos
            if n_pos > 0:
                weight = n_neg / n_pos
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
