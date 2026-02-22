"""
Inference script for VideoMAE CVS Classification.

Supports:
- Single video (folder of frames) inference
- Batch inference on multiple videos
- Sliding window inference over long videos
- Output predictions with confidence scores
"""
import os
import sys
import glob
import json
import argparse
import logging
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from transformers import VideoMAEImageProcessor

import config
from model import build_model
from utils import set_seed, setup_logging, load_checkpoint

logger = logging.getLogger("CVS_Classification")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with VideoMAE CVS Classifier")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to video frames directory or parent directory of multiple videos")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=os.path.join(config.RESULTS_DIR, "inference"),
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=config.CLASSIFICATION_THRESHOLD,
                        help="Classification threshold")
    parser.add_argument("--sliding_window", action="store_true",
                        help="Use sliding window for long videos")
    parser.add_argument("--window_stride", type=int, default=8,
                        help="Stride for sliding window (in frames)")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Process all subdirectories as separate videos")
    return parser.parse_args()


class CVSInferenceEngine:
    """
    Inference engine for CVS classification.
    
    Loads the trained VideoMAE model and processes video frames
    to predict CVS criteria achievement.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        threshold: float = config.CLASSIFICATION_THRESHOLD,
        device: torch.device = config.DEVICE,
    ):
        self.threshold = threshold
        self.device = device
        
        # Load processor
        self.processor = VideoMAEImageProcessor.from_pretrained(
            config.MODEL_NAME,
            size={"shortest_edge": config.IMAGE_SIZE},
            crop_size={"height": config.IMAGE_SIZE, "width": config.IMAGE_SIZE},
        )
        
        # Load model
        self.model = build_model()
        self.model, _, _ = load_checkpoint(self.model, checkpoint_path=checkpoint_path)
        self.model.eval()
        
        logger.info(f"Inference engine initialized (threshold={threshold})")
    
    def predict_single_clip(
        self,
        frame_paths: List[str],
        keyframe_idx: Optional[int] = None,
    ) -> Dict:
        """
        Predict CVS criteria for a single video clip.
        
        Args:
            frame_paths: Sorted list of frame file paths.
            keyframe_idx: Index of the center frame. If None, uses middle.
        
        Returns:
            Dict with probabilities, predictions, and CVS status.
        """
        if keyframe_idx is None:
            keyframe_idx = len(frame_paths) // 2
        
        # Sample frames
        indices = self._sample_indices(len(frame_paths), keyframe_idx)
        frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]
        
        # Process frames
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        
        probs = outputs["probabilities"].cpu().numpy()[0]
        preds = (probs >= self.threshold).astype(int)
        
        result = {
            "frame_index": keyframe_idx,
            "frame_path": frame_paths[keyframe_idx],
            "criteria": {},
            "cvs_achieved": bool(preds.sum() == config.NUM_CLASSES),
        }
        
        for i, name in enumerate(config.CVS_CRITERIA_NAMES):
            result["criteria"][name] = {
                "probability": float(probs[i]),
                "achieved": bool(preds[i]),
            }
        
        return result
    
    def predict_video(
        self,
        frames_dir: str,
        sliding_window: bool = False,
        window_stride: int = 8,
    ) -> Dict:
        """
        Predict CVS criteria for an entire video.
        
        Args:
            frames_dir: Directory containing video frames.
            sliding_window: If True, use sliding window approach.
            window_stride: Stride between windows.
        
        Returns:
            Dict with per-frame and aggregated predictions.
        """
        # Get sorted frame paths
        frame_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        frame_paths = []
        for ext in frame_exts:
            frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
        frame_paths = sorted(frame_paths)
        
        if len(frame_paths) == 0:
            logger.warning(f"No frames found in {frames_dir}")
            return {"error": "No frames found", "frames_dir": frames_dir}
        
        logger.info(f"Processing video: {frames_dir} ({len(frame_paths)} frames)")
        
        if sliding_window:
            return self._sliding_window_predict(frame_paths, window_stride)
        else:
            # Single clip prediction from the middle of the video
            return self.predict_single_clip(frame_paths)
    
    def _sliding_window_predict(
        self,
        frame_paths: List[str],
        stride: int = 8,
    ) -> Dict:
        """
        Sliding window inference over a long video.
        Aggregates predictions across all windows.
        """
        total_frames = len(frame_paths)
        # Generate window centers
        clip_length = config.NUM_FRAMES * config.FRAME_SAMPLE_RATE
        centers = list(range(clip_length // 2, total_frames - clip_length // 2, stride))
        if not centers:
            centers = [total_frames // 2]
        
        all_probs = []
        window_results = []
        
        for center in tqdm(centers, desc="Sliding window", leave=False):
            result = self.predict_single_clip(frame_paths, keyframe_idx=center)
            probs = [result["criteria"][name]["probability"] for name in config.CVS_CRITERIA_NAMES]
            all_probs.append(probs)
            window_results.append(result)
        
        # Aggregate (mean probabilities)
        mean_probs = np.mean(all_probs, axis=0)
        agg_preds = (mean_probs >= self.threshold).astype(int)
        
        aggregated = {
            "frames_dir": os.path.basename(os.path.dirname(frame_paths[0]) if frame_paths else ""),
            "num_frames": total_frames,
            "num_windows": len(centers),
            "stride": stride,
            "criteria": {},
            "cvs_achieved": bool(agg_preds.sum() == config.NUM_CLASSES),
            "window_results": window_results,
        }
        
        for i, name in enumerate(config.CVS_CRITERIA_NAMES):
            aggregated["criteria"][name] = {
                "mean_probability": float(mean_probs[i]),
                "achieved": bool(agg_preds[i]),
                "min_probability": float(np.min([p[i] for p in all_probs])),
                "max_probability": float(np.max([p[i] for p in all_probs])),
                "std_probability": float(np.std([p[i] for p in all_probs])),
            }
        
        return aggregated
    
    def _sample_indices(self, total_frames: int, keyframe_idx: int) -> List[int]:
        """Sample frame indices for a clip centered around keyframe."""
        clip_len = config.NUM_FRAMES * config.FRAME_SAMPLE_RATE
        half_clip = clip_len // 2
        start = keyframe_idx - half_clip
        
        indices = []
        for i in range(config.NUM_FRAMES):
            idx = start + i * config.FRAME_SAMPLE_RATE
            idx = max(0, min(idx, total_frames - 1))
            indices.append(idx)
        
        return indices


def run_inference(args):
    """Main inference function."""
    set_seed()
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("  VideoMAE CVS Classification — Inference")
    logger.info("=" * 70)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Create inference engine
    engine = CVSInferenceEngine(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    
    if args.batch_mode:
        # Process all subdirectories
        subdirs = sorted([
            d for d in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, d))
        ])
        
        all_results = {}
        for subdir in subdirs:
            frames_dir = os.path.join(args.input, subdir)
            result = engine.predict_video(
                frames_dir=frames_dir,
                sliding_window=args.sliding_window,
                window_stride=args.window_stride,
            )
            all_results[subdir] = result
            
            # Print per-video result
            cvs_status = "✅ ACHIEVED" if result.get("cvs_achieved", False) else "❌ NOT ACHIEVED"
            print(f"\n  Video: {subdir} — CVS: {cvs_status}")
            for name, data in result.get("criteria", {}).items():
                prob = data.get("probability", data.get("mean_probability", 0))
                achieved = data.get("achieved", False)
                status = "✓" if achieved else "✗"
                print(f"    {status} {name}: {prob:.3f}")
        
        # Save batch results
        output_path = os.path.join(args.output, "batch_predictions.json")
        with open(output_path, "w") as f:
            # Remove window_results for JSON serialization (too large)
            serializable = {}
            for k, v in all_results.items():
                v_copy = {kk: vv for kk, vv in v.items() if kk != "window_results"}
                serializable[k] = v_copy
            json.dump(serializable, f, indent=2)
        
        logger.info(f"\nBatch results saved to: {output_path}")
    else:
        # Single video inference
        result = engine.predict_video(
            frames_dir=args.input,
            sliding_window=args.sliding_window,
            window_stride=args.window_stride,
        )
        
        # Print result
        cvs_status = "✅ ACHIEVED" if result.get("cvs_achieved", False) else "❌ NOT ACHIEVED"
        print(f"\n{'=' * 50}")
        print(f"  CVS Assessment: {cvs_status}")
        print(f"{'=' * 50}")
        for name, data in result.get("criteria", {}).items():
            prob = data.get("probability", data.get("mean_probability", 0))
            achieved = data.get("achieved", False)
            status = "✅" if achieved else "❌"
            print(f"  {status} {name}: {prob:.4f}")
        print()
        
        # Save result
        output_path = os.path.join(args.output, "prediction.json")
        result_serializable = {k: v for k, v in result.items() if k != "window_results"}
        with open(output_path, "w") as f:
            json.dump(result_serializable, f, indent=2)
        
        logger.info(f"Result saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
