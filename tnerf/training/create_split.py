"""
Data Split Script for T-NeRF Training.

This script splits the dataset into train/val/test sets with 8:1:1 ratio.
For 2000 total samples: 1600 train, 200 val, 200 test.

Usage:
    python create_split.py --data_dir /path/to/tnerf/data
    python create_split.py --data_dir /path/to/tnerf/data --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
    python create_split.py --data_dir "/media/fengwu/ZX1 1TB/code/cv_finalproject/data/tnerf" --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Split T-NeRF dataset into train/val/test")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to T-NeRF data directory (containing samples/)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio for training set (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio for validation set (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio for test set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing split files")
    
    return parser.parse_args()


def discover_samples(data_dir: Path) -> List[str]:
    """
    Discover all valid samples in the data directory.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        List of sample IDs (directory names)
    """
    samples_dir = data_dir / "samples"
    
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    # Find all subdirectories with valid data
    sample_ids = []
    
    for sample_path in sorted(samples_dir.iterdir()):
        if not sample_path.is_dir():
            continue
        
        # Check if this is a valid sample (has images/ directory)
        images_dir = sample_path / "images"
        if images_dir.exists() and any(images_dir.glob("view_*.png")):
            sample_ids.append(sample_path.name)
    
    return sample_ids


def split_samples(
    sample_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split sample IDs into train/val/test sets.
    
    Args:
        sample_ids: List of all sample IDs
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Shuffle samples
    random.seed(seed)
    shuffled = sample_ids.copy()
    random.shuffle(shuffled)
    
    # Calculate split sizes
    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val  # Remainder goes to test
    
    # Split
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def create_split_file(
    sample_ids: List[str],
    output_path: Path,
    split_name: str
) -> Dict:
    """
    Create a split JSON file.
    
    Args:
        sample_ids: List of sample IDs for this split
        output_path: Path to save the JSON file
        split_name: Name of the split (train/val/test)
        
    Returns:
        The data dictionary that was saved
    """
    # Create sample entries
    samples = [{"sample_id": sid} for sid in sorted(sample_ids)]
    
    data = {
        "split": split_name,
        "num_samples": len(samples),
        "samples": samples
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return data


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    
    print(f"Data directory: {data_dir}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    
    # Check for existing split files
    train_path = data_dir / "train_samples.json"
    val_path = data_dir / "val_samples.json"
    test_path = data_dir / "test_samples.json"
    
    if not args.overwrite:
        existing = []
        if train_path.exists():
            existing.append("train_samples.json")
        if val_path.exists():
            existing.append("val_samples.json")
        if test_path.exists():
            existing.append("test_samples.json")
        
        if existing:
            print(f"\nWarning: The following split files already exist:")
            for f in existing:
                print(f"  - {f}")
            print("\nUse --overwrite to replace them, or delete them manually.")
            
            response = input("Continue and overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
    
    # Discover all samples
    print("\nDiscovering samples...")
    sample_ids = discover_samples(data_dir)
    print(f"Found {len(sample_ids)} samples")
    
    if len(sample_ids) == 0:
        print("Error: No valid samples found!")
        return
    
    # Split samples
    print("\nSplitting samples...")
    train_ids, val_ids, test_ids = split_samples(
        sample_ids,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"  Train: {len(train_ids)} samples ({len(train_ids)/len(sample_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} samples ({len(val_ids)/len(sample_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} samples ({len(test_ids)/len(sample_ids)*100:.1f}%)")
    
    # Create split files
    print("\nCreating split files...")
    
    create_split_file(train_ids, train_path, "train")
    print(f"  Created: {train_path}")
    
    create_split_file(val_ids, val_path, "val")
    print(f"  Created: {val_path}")
    
    create_split_file(test_ids, test_path, "test")
    print(f"  Created: {test_path}")
    
    # Also create a combined config file
    config_path = data_dir / "split_config.json"
    config = {
        "total_samples": len(sample_ids),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Created: {config_path}")
    
    print("\nDone!")
    print("\nSummary:")
    print(f"  Total samples: {len(sample_ids)}")
    print(f"  Train: {len(train_ids)} ({args.train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_ids)} ({args.val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_ids)} ({args.test_ratio*100:.0f}%)")


if __name__ == "__main__":
    main()
