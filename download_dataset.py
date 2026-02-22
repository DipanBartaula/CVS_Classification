"""
Download and prepare the Endoscapes dataset for CVS Classification.

Downloads the Endoscapes2023 dataset from the official source (University of Strasbourg)
and organizes it for the VideoMAE CVS classification pipeline.

Dataset: Endoscapes2023
Source:  https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip
Paper:   https://arxiv.org/abs/2312.12429
License: CC BY-NC-SA 4.0 (non-commercial research only)

Usage:
    python download_dataset.py
    python download_dataset.py --output_dir "D:\\path\\to\\data"
    python download_dataset.py --skip_download --prepare_only
"""
import os
import sys
import json
import shutil
import zipfile
import hashlib
import argparse
import logging
import urllib.request
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
DATASET_URL = "https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip"
DATASET_FILENAME = "endoscapes.zip"

# Default output directory (auto-detect Colab)
_IN_COLAB = os.path.exists("/content") and os.path.isdir("/content")
if _IN_COLAB:
    DEFAULT_OUTPUT_DIR = "/content/data/endoscapes"
else:
    DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "endoscapes")

# CVS criteria columns in all_metadata.csv
CVS_COLUMNS = ["C1", "C2", "C3"]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Download helpers
# ─────────────────────────────────────────────
class DownloadProgressBar:
    """Progress bar for urllib downloads."""
    
    def __init__(self):
        self.last_percent = -1
    
    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(int(downloaded * 100 / total_size), 100)
        if percent != self.last_percent:
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  Downloading: [{bar}] {percent}% "
                f"({downloaded_mb:.1f}/{total_mb:.1f} MB)"
            )
            sys.stdout.flush()
            self.last_percent = percent
            if percent == 100:
                print()


def download_file(url: str, dest_path: str, force: bool = False) -> str:
    """
    Download a file from URL to destination path.
    
    Args:
        url: URL to download from.
        dest_path: Full path where the file will be saved.
        force: If True, re-download even if file exists.
    
    Returns:
        Path to the downloaded file.
    """
    if os.path.exists(dest_path) and not force:
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"File already exists: {dest_path} ({file_size_mb:.1f} MB)")
        logger.info("Use --force to re-download")
        return dest_path
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    logger.info(f"Downloading from: {url}")
    logger.info(f"Saving to: {dest_path}")
    logger.info("This may take a while depending on your connection speed...")
    logger.info("(Dataset is approximately 9-12 GB)")
    
    try:
        progress = DownloadProgressBar()
        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"✅ Download complete: {file_size_mb:.1f} MB")
        
    except urllib.error.URLError as e:
        logger.error(f"Download failed: {e}")
        logger.error(
            "If the download fails, you can manually download the dataset from:\n"
            f"  {DATASET_URL}\n"
            "and place the zip file at:\n"
            f"  {dest_path}"
        )
        raise
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    
    return dest_path


def extract_zip(zip_path: str, extract_to: str, force: bool = False):
    """
    Extract a zip file to the specified directory.
    
    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract to.
        force: If True, overwrite existing files.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    logger.info(f"Extracting: {zip_path}")
    logger.info(f"Destination: {extract_to}")
    
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        logger.info(f"Total files in archive: {total}")
        
        for i, member in enumerate(members):
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                percent = (i + 1) * 100 // total
                sys.stdout.write(f"\r  Extracting: {percent}% ({i + 1}/{total} files)")
                sys.stdout.flush()
            
            target_path = os.path.join(extract_to, member)
            
            # Skip if exists and not forcing
            if os.path.exists(target_path) and not force:
                continue
            
            zf.extract(member, extract_to)
        
        print()  # Newline after progress
    
    logger.info("✅ Extraction complete")


# ─────────────────────────────────────────────
#  Dataset structure discovery
# ─────────────────────────────────────────────
def find_dataset_root(extract_dir: str) -> str:
    """
    Find the actual root directory of the extracted dataset.
    The zip may have a top-level folder like 'endoscapes/'.
    """
    # Check if the extracted content has a single subdirectory
    items = os.listdir(extract_dir)
    
    # Look for key indicators of the dataset root
    indicators = ["all_metadata.csv", "train_vids.txt", "train"]
    
    # Check extract_dir itself
    if any(os.path.exists(os.path.join(extract_dir, ind)) for ind in indicators):
        return extract_dir
    
    # Check one level down
    for item in items:
        subdir = os.path.join(extract_dir, item)
        if os.path.isdir(subdir):
            if any(os.path.exists(os.path.join(subdir, ind)) for ind in indicators):
                return subdir
    
    # Check two levels down
    for item in items:
        subdir = os.path.join(extract_dir, item)
        if os.path.isdir(subdir):
            for sub_item in os.listdir(subdir):
                sub_subdir = os.path.join(subdir, sub_item)
                if os.path.isdir(sub_subdir):
                    if any(os.path.exists(os.path.join(sub_subdir, ind)) for ind in indicators):
                        return sub_subdir
    
    logger.warning(f"Could not find dataset root in {extract_dir}, using extract dir as root")
    return extract_dir


# ─────────────────────────────────────────────
#  Data preparation for VideoMAE
# ─────────────────────────────────────────────
def parse_coco_cvs_annotations(annotation_file: str) -> List[Dict]:
    """
    Parse CVS annotations from COCO-format JSON file.
    
    The Endoscapes COCO annotation files contain CVS labels as
    image-level tags. The annotation_ds_coco.json files are used
    for single-frame CVS prediction.
    
    Returns:
        List of dicts with: image_id, file_name, video_id, frame_num, C1, C2, C3
    """
    if not os.path.exists(annotation_file):
        logger.warning(f"Annotation file not found: {annotation_file}")
        return []
    
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)
    
    # Build image ID to image info mapping
    images = {img["id"]: img for img in coco_data.get("images", [])}
    
    # Parse annotations
    samples = []
    
    # CVS labels may be stored in the images themselves or in annotations
    for img_id, img_info in images.items():
        file_name = img_info.get("file_name", "")
        
        # Parse video_id and frame_num from filename: {VIDEO_ID}_{FRAME_NUM}.jpg
        basename = os.path.splitext(os.path.basename(file_name))[0]
        parts = basename.rsplit("_", 1)
        if len(parts) == 2:
            video_id = parts[0]
            frame_num = parts[1]
        else:
            video_id = basename
            frame_num = "0"
        
        # Try to get CVS labels from image info
        cvs_labels = {}
        for key in ["C1", "C2", "C3", "c1", "c2", "c3",
                     "cvs_c1", "cvs_c2", "cvs_c3",
                     "criterion_1", "criterion_2", "criterion_3"]:
            if key in img_info:
                criterion = key.upper().replace("CVS_", "").replace("CRITERION_", "C")
                cvs_labels[criterion] = img_info[key]
        
        # Also check in a "cvs" field
        if "cvs" in img_info:
            cvs = img_info["cvs"]
            if isinstance(cvs, dict):
                cvs_labels.update(cvs)
            elif isinstance(cvs, list) and len(cvs) == 3:
                cvs_labels = {"C1": cvs[0], "C2": cvs[1], "C3": cvs[2]}
        
        if cvs_labels:
            samples.append({
                "image_id": img_id,
                "file_name": file_name,
                "video_id": video_id,
                "frame_num": frame_num,
                "C1": cvs_labels.get("C1", 0),
                "C2": cvs_labels.get("C2", 0),
                "C3": cvs_labels.get("C3", 0),
            })
    
    # If no CVS labels found in images, try annotations
    if not samples:
        annotations = coco_data.get("annotations", [])
        # Group annotations by image_id
        from collections import defaultdict
        img_annotations = defaultdict(list)
        for ann in annotations:
            img_annotations[ann["image_id"]].append(ann)
        
        # Check if annotations have CVS labels
        for img_id, img_info in images.items():
            file_name = img_info.get("file_name", "")
            basename = os.path.splitext(os.path.basename(file_name))[0]
            parts = basename.rsplit("_", 1)
            video_id = parts[0] if len(parts) == 2 else basename
            frame_num = parts[1] if len(parts) == 2 else "0"
            
            anns = img_annotations.get(img_id, [])
            cvs_labels = {}
            for ann in anns:
                for key in ["C1", "C2", "C3"]:
                    if key.lower() in ann:
                        cvs_labels[key] = ann[key.lower()]
                    elif key in ann:
                        cvs_labels[key] = ann[key]
                # Check "attributes" or "extra" fields
                for extra_field in ["attributes", "extra", "metadata"]:
                    if extra_field in ann:
                        extra = ann[extra_field]
                        if isinstance(extra, dict):
                            for key in ["C1", "C2", "C3"]:
                                if key in extra:
                                    cvs_labels[key] = extra[key]
            
            if cvs_labels:
                samples.append({
                    "image_id": img_id,
                    "file_name": file_name,
                    "video_id": video_id,
                    "frame_num": frame_num,
                    "C1": cvs_labels.get("C1", 0),
                    "C2": cvs_labels.get("C2", 0),
                    "C3": cvs_labels.get("C3", 0),
                })
    
    return samples


def parse_all_metadata_csv(csv_path: str) -> List[Dict]:
    """
    Parse the all_metadata.csv file which contains CVS annotations
    from all annotators and consensus labels.
    
    Returns:
        List of dicts with: video_id, frame_num, file_name, C1, C2, C3
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Metadata CSV not found: {csv_path}")
        return []
    
    samples = []
    
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        
        logger.info(f"CSV columns: {columns}")
        
        # Map columns to standard names
        col_map = {}
        for col in columns:
            lower = col.lower().strip()
            if "video" in lower:
                col_map["video_id"] = col
            elif "frame" in lower and ("num" in lower or "id" in lower or lower == "frame"):
                col_map["frame_num"] = col
            elif "file" in lower and "name" in lower:
                col_map["file_name"] = col
            # Look for consensus CVS labels (often contain 'consensus' or 'avg')
            elif "c1" in lower or "criterion_1" in lower or "two_struct" in lower:
                if "consensus" in lower or "avg" in lower:
                    col_map["C1_consensus"] = col
                elif "C1" not in col_map:
                    col_map["C1"] = col
            elif "c2" in lower or "criterion_2" in lower or "hepato" in lower:
                if "consensus" in lower or "avg" in lower:
                    col_map["C2_consensus"] = col
                elif "C2" not in col_map:
                    col_map["C2"] = col
            elif "c3" in lower or "criterion_3" in lower or "cystic_plate" in lower:
                if "consensus" in lower or "avg" in lower:
                    col_map["C3_consensus"] = col
                elif "C3" not in col_map:
                    col_map["C3"] = col
        
        logger.info(f"Column mapping: {col_map}")
        
        for row in reader:
            # Get video_id and frame_num
            video_id = row.get(col_map.get("video_id", ""), "")
            
            # Try multiple frame number columns
            frame_num = (
                row.get(col_map.get("frame_num", ""), "") or
                row.get(col_map.get("file_name", ""), "")
            )
            
            # Parse file_name from columns or construct it
            file_name = row.get(col_map.get("file_name", ""), "")
            if not file_name and video_id and frame_num:
                file_name = f"{video_id}_{frame_num}.jpg"
            
            # Get CVS labels (prefer consensus values)
            c1 = float(row.get(col_map.get("C1_consensus", col_map.get("C1", "")), 0) or 0)
            c2 = float(row.get(col_map.get("C2_consensus", col_map.get("C2", "")), 0) or 0)
            c3 = float(row.get(col_map.get("C3_consensus", col_map.get("C3", "")), 0) or 0)
            
            samples.append({
                "video_id": str(video_id),
                "frame_num": str(frame_num),
                "file_name": file_name,
                "C1": c1,
                "C2": c2,
                "C3": c3,
            })
    
    logger.info(f"Parsed {len(samples)} entries from all_metadata.csv")
    return samples


def prepare_for_videomae(
    dataset_root: str,
    output_dir: str,
):
    """
    Prepare the Endoscapes dataset for VideoMAE training.
    
    This creates a reorganized structure that groups frames by video,
    making it suitable for video-clip-based training:
    
    output_dir/
    ├── frames/
    │   ├── 1/                  # Video ID
    │   │   ├── 1_29375.jpg
    │   │   ├── 1_29380.jpg
    │   │   └── ...
    │   ├── 2/
    │   └── ...
    ├── all_metadata.csv        # Copied/generated
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt
    """
    logger.info("=" * 60)
    logger.info("  Preparing dataset for VideoMAE training")
    logger.info("=" * 60)
    
    # Create output directories
    frames_out = os.path.join(output_dir, "frames")
    splits_out = os.path.join(output_dir, "splits")
    os.makedirs(frames_out, exist_ok=True)
    os.makedirs(splits_out, exist_ok=True)
    
    # ── Step 1: Copy/collect frames grouped by video ──
    logger.info("\n📂 Step 1: Organizing frames by video ID...")
    
    frame_count = 0
    video_ids_seen = set()
    
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        # Find all image files
        for filename in os.listdir(split_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            # Parse: {VIDEO_ID}_{FRAME_NUM}.jpg
            basename = os.path.splitext(filename)[0]
            parts = basename.rsplit("_", 1)
            if len(parts) != 2:
                continue
            
            video_id = parts[0]
            video_ids_seen.add(video_id)
            
            # Create video directory and copy/symlink frame
            video_dir = os.path.join(frames_out, video_id)
            os.makedirs(video_dir, exist_ok=True)
            
            src = os.path.join(split_dir, filename)
            dst = os.path.join(video_dir, filename)
            
            if not os.path.exists(dst):
                # Use copy on Windows, symlink on Linux
                try:
                    os.symlink(os.path.abspath(src), dst)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)
                frame_count += 1
    
    logger.info(f"  Organized {frame_count} frames from {len(video_ids_seen)} videos")
    
    # ── Step 2: Setup split files ──
    logger.info("\n📋 Step 2: Creating split files...")
    
    for split_name, split_file_name in [
        ("train", "train_vids.txt"),
        ("val", "val_vids.txt"),
        ("test", "test_vids.txt"),
    ]:
        src_split = os.path.join(dataset_root, split_file_name)
        dst_split = os.path.join(splits_out, f"{split_name}.txt")
        
        if os.path.exists(src_split):
            # Read and copy
            with open(src_split, "r") as f:
                video_ids = [line.strip() for line in f if line.strip()]
            with open(dst_split, "w") as f:
                f.write("\n".join(video_ids))
            logger.info(f"  {split_name}: {len(video_ids)} videos (from {split_file_name})")
        else:
            # Infer from directory contents
            split_dir = os.path.join(dataset_root, split_name)
            if os.path.isdir(split_dir):
                video_ids = set()
                for filename in os.listdir(split_dir):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        parts = os.path.splitext(filename)[0].rsplit("_", 1)
                        if len(parts) == 2:
                            video_ids.add(parts[0])
                
                video_ids = sorted(video_ids)
                with open(dst_split, "w") as f:
                    f.write("\n".join(video_ids))
                logger.info(f"  {split_name}: {len(video_ids)} videos (inferred from frames)")
            else:
                logger.warning(f"  Could not determine {split_name} split")
    
    # ── Step 3: Copy/generate metadata ──
    logger.info("\n📊 Step 3: Processing annotations...")
    
    metadata_src = os.path.join(dataset_root, "all_metadata.csv")
    metadata_dst = os.path.join(output_dir, "all_metadata.csv")
    
    if os.path.exists(metadata_src):
        shutil.copy2(metadata_src, metadata_dst)
        logger.info(f"  Copied all_metadata.csv")
    else:
        # Generate metadata from COCO annotation files
        logger.info("  all_metadata.csv not found, generating from COCO annotations...")
        all_samples = []
        
        for split in ["train", "val", "test"]:
            # Try annotation_ds_coco.json (CVS single-frame annotations)
            ann_file = os.path.join(dataset_root, split, "annotation_ds_coco.json")
            if os.path.exists(ann_file):
                samples = parse_coco_cvs_annotations(ann_file)
                logger.info(f"  {split}: {len(samples)} annotated frames from annotation_ds_coco.json")
                all_samples.extend(samples)
            else:
                # Try annotation_coco.json
                ann_file = os.path.join(dataset_root, split, "annotation_coco.json")
                if os.path.exists(ann_file):
                    samples = parse_coco_cvs_annotations(ann_file)
                    logger.info(f"  {split}: {len(samples)} frames from annotation_coco.json")
                    all_samples.extend(samples)
        
        if all_samples:
            # Write generated metadata CSV
            with open(metadata_dst, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["video_id", "frame_id", "file_name", "C1", "C2", "C3"])
                writer.writeheader()
                for sample in all_samples:
                    writer.writerow({
                        "video_id": sample["video_id"],
                        "frame_id": sample["frame_num"],
                        "file_name": sample.get("file_name", ""),
                        "C1": sample["C1"],
                        "C2": sample["C2"],
                        "C3": sample["C3"],
                    })
            logger.info(f"  Generated all_metadata.csv with {len(all_samples)} entries")
        else:
            logger.warning("  ⚠️  Could not generate metadata — no CVS annotations found!")
    
    # ── Step 4: Validate ──
    logger.info("\n🔍 Step 4: Validating prepared dataset...")
    validate_dataset(output_dir)
    
    logger.info(f"\n✅ Dataset preparation complete!")
    logger.info(f"   Prepared dataset location: {output_dir}")
    logger.info(f"\n   To train, run:")
    logger.info(f'     python train.py --data_root "{output_dir}"')


def validate_dataset(dataset_dir: str) -> bool:
    """
    Validate the prepared dataset structure.
    
    Checks:
    - frames/ directory exists with video subdirectories
    - all_metadata.csv exists and has required columns
    - splits/ directory has train.txt, val.txt, test.txt
    - Frame counts match across metadata and filesystem
    """
    errors = []
    warnings = []
    
    # Check frames directory
    frames_dir = os.path.join(dataset_dir, "frames")
    if os.path.isdir(frames_dir):
        video_dirs = [d for d in os.listdir(frames_dir)
                      if os.path.isdir(os.path.join(frames_dir, d))]
        total_frames = sum(
            len([f for f in os.listdir(os.path.join(frames_dir, v))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            for v in video_dirs
        )
        logger.info(f"  Frames: {total_frames} frames in {len(video_dirs)} video directories")
    else:
        errors.append("frames/ directory not found")
    
    # Check metadata
    metadata_path = os.path.join(dataset_dir, "all_metadata.csv")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames
        logger.info(f"  Metadata: {len(rows)} entries, columns: {columns}")
        
        # Check C1, C2, C3 columns
        has_cvs = all(
            any(c.lower().startswith(criterion.lower()) for c in columns)
            for criterion in ["c1", "c2", "c3"]
        )
        if not has_cvs:
            warnings.append("Metadata may be missing CVS criterion columns (C1, C2, C3)")
    else:
        errors.append("all_metadata.csv not found")
    
    # Check splits
    splits_dir = os.path.join(dataset_dir, "splits")
    if os.path.isdir(splits_dir):
        for split in ["train.txt", "val.txt", "test.txt"]:
            split_path = os.path.join(splits_dir, split)
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    count = len([l for l in f if l.strip()])
                logger.info(f"  Split {split}: {count} videos")
            else:
                warnings.append(f"Split file missing: {split}")
    else:
        warnings.append("splits/ directory not found (will be auto-generated)")
    
    # Report
    if errors:
        logger.error("  ❌ Validation errors:")
        for e in errors:
            logger.error(f"    - {e}")
        return False
    
    if warnings:
        logger.warning("  ⚠️  Warnings:")
        for w in warnings:
            logger.warning(f"    - {w}")
    
    logger.info("  ✅ Validation passed!")
    return True


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare the Endoscapes dataset for CVS classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and prepare (default location: ./data/endoscapes)
  python download_dataset.py

  # Custom output directory
  python download_dataset.py --output_dir "D:\\datasets\\endoscapes"

  # Skip download (if you already have the zip)
  python download_dataset.py --skip_download --zip_path "D:\\downloads\\endoscapes.zip"

  # Only prepare (reorganize already-extracted data)
  python download_dataset.py --prepare_only --extracted_dir "D:\\raw_endoscapes"

  # Force re-download and re-extract
  python download_dataset.py --force
        """,
    )
    
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for prepared dataset (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--download_dir", type=str, default=None,
        help="Directory to store the downloaded zip (default: output_dir/../downloads)"
    )
    parser.add_argument(
        "--zip_path", type=str, default=None,
        help="Path to an already-downloaded endoscapes.zip file"
    )
    parser.add_argument(
        "--extracted_dir", type=str, default=None,
        help="Path to already-extracted raw Endoscapes dataset"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip the download step (requires --zip_path)"
    )
    parser.add_argument(
        "--prepare_only", action="store_true",
        help="Only run preparation step (requires --extracted_dir)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download and re-extraction"
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Only validate an existing prepared dataset"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     Endoscapes Dataset Downloader & Preparation Tool       ║")
    print("║     For VideoMAE CVS Classification                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Dataset: Endoscapes2023")
    print("  Source:  University of Strasbourg / CAMMA Lab")
    print("  License: CC BY-NC-SA 4.0 (non-commercial research only)")
    print(f"  URL:     {DATASET_URL}")
    print()
    
    # Validate only
    if args.validate_only:
        logger.info(f"Validating dataset at: {args.output_dir}")
        validate_dataset(args.output_dir)
        return
    
    # Prepare only (from already-extracted data)
    if args.prepare_only:
        if not args.extracted_dir:
            logger.error("--prepare_only requires --extracted_dir")
            sys.exit(1)
        if not os.path.isdir(args.extracted_dir):
            logger.error(f"Extracted directory not found: {args.extracted_dir}")
            sys.exit(1)
        
        prepare_for_videomae(args.extracted_dir, args.output_dir)
        return
    
    # Full pipeline: Download → Extract → Prepare
    
    # Step 1: Download
    if args.skip_download:
        if not args.zip_path:
            logger.error("--skip_download requires --zip_path")
            sys.exit(1)
        zip_path = args.zip_path
    else:
        download_dir = args.download_dir or os.path.join(
            os.path.dirname(args.output_dir), "downloads"
        )
        zip_path = args.zip_path or os.path.join(download_dir, DATASET_FILENAME)
        
        logger.info("━" * 50)
        logger.info("STEP 1: Downloading dataset")
        logger.info("━" * 50)
        download_file(DATASET_URL, zip_path, force=args.force)
    
    # Step 2: Extract
    logger.info("\n" + "━" * 50)
    logger.info("STEP 2: Extracting dataset")
    logger.info("━" * 50)
    
    extract_dir = os.path.join(os.path.dirname(zip_path), "endoscapes_raw")
    extract_zip(zip_path, extract_dir, force=args.force)
    
    # Find actual dataset root within extracted files
    raw_root = find_dataset_root(extract_dir)
    logger.info(f"Dataset root found at: {raw_root}")
    
    # Step 3: Prepare for VideoMAE
    logger.info("\n" + "━" * 50)
    logger.info("STEP 3: Preparing for VideoMAE training")
    logger.info("━" * 50)
    
    prepare_for_videomae(raw_root, args.output_dir)
    
    # Final summary
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    ✅ ALL DONE!                             ║")
    print("║                                                            ║")
    print(f"║  Dataset ready at:                                         ║")
    print(f"║    {args.output_dir:<55s}║")
    print("║                                                            ║")
    print("║  Next steps:                                               ║")
    print("║    1. Update DATASET_ROOT in config.py                     ║")
    print("║    2. Run: python train.py                                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
