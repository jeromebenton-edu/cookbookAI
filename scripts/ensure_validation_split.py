#!/usr/bin/env python3
"""Ensure dataset has a non-empty validation split.

This script checks if the dataset has a validation split and creates one if missing or empty.
Used to guarantee training metrics work correctly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from datasets import DatasetDict, load_from_disk

LOG = logging.getLogger(__name__)


def create_validation_split_standalone(ds_dict: DatasetDict, val_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Create a validation split from train if one doesn't exist.
    Standalone version without dependencies on training modules.
    """
    # Check for existing validation split
    existing_val_split = None
    if "validation" in ds_dict:
        existing_val_split = "validation"
    elif "val" in ds_dict:
        existing_val_split = "val"
    elif "eval" in ds_dict:
        existing_val_split = "eval"

    # If validation exists and is non-empty, we're good
    if existing_val_split:
        val_size = len(ds_dict[existing_val_split])
        if val_size > 0:
            print(f"Validation split '{existing_val_split}' already exists with {val_size} examples")
            return ds_dict
        else:
            print(f"Validation split '{existing_val_split}' exists but is EMPTY, will recreate")
            splits_to_keep = {k: v for k, v in ds_dict.items() if k != existing_val_split}
            ds_dict = DatasetDict(splits_to_keep)

    if "train" not in ds_dict:
        raise RuntimeError("No train split found, cannot create validation split")

    train_size = len(ds_dict["train"])

    if train_size < 10:
        raise RuntimeError(
            f"Training set too small ({train_size} examples) to create validation split. "
            f"Need at least 10 examples."
        )

    print(f"Creating {val_ratio:.0%} validation split from train (seed={seed})")
    print(f"  Train size before split: {train_size}")

    train_ds = ds_dict["train"]

    # Split the train dataset
    split_ds = train_ds.train_test_split(test_size=val_ratio, seed=seed)

    # Create new DatasetDict with standardized "validation" name
    new_dict = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"]
    })

    # Copy over any other splits
    for split_name, split_data in ds_dict.items():
        if split_name not in new_dict and split_name != "train":
            new_dict[split_name] = split_data

    train_new = len(new_dict['train'])
    val_new = len(new_dict['validation'])

    # Verify splits
    if val_new == 0:
        raise RuntimeError(f"Validation split is empty after creation!")

    if train_new == 0:
        raise RuntimeError(f"Training split is empty after creation!")

    print(f"✓ Split complete: train={train_new}, validation={val_new}")
    print(f"  Validation ratio: {val_new / train_size:.1%} of original train set")

    return new_dict


def save_dataset_with_validation_standalone(ds_dict: DatasetDict, dataset_dir: Path) -> None:
    """Save dataset with validation split, standalone version."""
    import shutil
    import tempfile

    if "validation" not in ds_dict:
        raise RuntimeError(f"Cannot save dataset: no 'validation' split found")

    val_size = len(ds_dict["validation"])
    if val_size == 0:
        raise RuntimeError(f"Cannot save dataset: validation split is EMPTY")

    save_path = dataset_dir / "dataset_dict" if dataset_dir.name != "dataset_dict" else dataset_dir

    print(f"Saving dataset with validation split to {save_path}")
    print(f"  Splits: {', '.join(f'{k}={len(v)}' for k, v in ds_dict.items())}")

    # Save to temporary location first, then replace
    temp_parent = save_path.parent
    temp_name = f"dataset_dict.new.{int(time.time())}"
    temp_path = temp_parent / temp_name

    try:
        # Save to temp location
        print(f"  Saving to temporary location: {temp_path}")
        ds_dict.save_to_disk(str(temp_path))

        # Verify temp save
        loaded_temp = load_from_disk(str(temp_path))
        temp_val_size = len(loaded_temp["validation"]) if "validation" in loaded_temp else 0
        if temp_val_size == 0:
            raise RuntimeError(f"Validation split is empty in temp save!")

        print(f"  Temp save verified: validation_size={temp_val_size}")

        # Create backup of original
        if save_path.exists():
            backup_name = f"dataset_dict.backup.{int(time.time())}"
            backup_path = temp_parent / backup_name
            print(f"  Creating backup: {backup_path}")
            shutil.move(str(save_path), str(backup_path))

        # Move temp to final location
        print(f"  Moving to final location: {save_path}")
        shutil.move(str(temp_path), str(save_path))

        # Final verification
        if not save_path.exists():
            raise RuntimeError(f"Dataset save failed: {save_path} does not exist after save")

        loaded = load_from_disk(str(save_path))
        loaded_val_size = len(loaded["validation"]) if "validation" in loaded else 0
        if loaded_val_size == 0:
            raise RuntimeError(f"Validation split is empty after final save!")

        print(f"✓ Dataset saved and verified: validation_size={loaded_val_size}")

    except Exception as e:
        # Cleanup temp on failure
        if temp_path.exists():
            shutil.rmtree(temp_path)
        print(f"❌ Save failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Ensure dataset has a non-empty validation split"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/datasets/boston_layoutlmv3_full"),
        help="Dataset directory (contains dataset_dict folder)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio if creating new split (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split (default: 42)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of validation split even if non-empty one exists"
    )

    args = parser.parse_args()

    dataset_path = args.dataset_dir / "dataset_dict"

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print(f"   Run: make build-dataset")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}")
    ds = load_from_disk(str(dataset_path))

    if not isinstance(ds, DatasetDict):
        print(f"❌ Error: Dataset is not a DatasetDict (type: {type(ds).__name__})")
        sys.exit(1)

    print(f"Current splits: {', '.join(f'{k}={len(v)}' for k, v in ds.items())}")

    # Check validation status
    val_key = None
    if "validation" in ds:
        val_key = "validation"
    elif "val" in ds:
        val_key = "val"
    elif "eval" in ds:
        val_key = "eval"

    needs_split = False

    if val_key is None:
        print("❌ No validation split found")
        needs_split = True
    else:
        val_size = len(ds[val_key])
        if val_size == 0:
            print(f"❌ Validation split '{val_key}' exists but is EMPTY (0 examples)")
            needs_split = True
        elif args.force:
            print(f"⚠️  Validation split '{val_key}' exists with {val_size} examples, but --force specified")
            needs_split = True
        else:
            print(f"✓ Validation split '{val_key}' exists with {val_size} examples")
            print("Nothing to do (use --force to recreate)")
            sys.exit(0)

    if needs_split:
        print(f"\nCreating {args.val_ratio:.0%} validation split (seed={args.seed})...")

        try:
            ds_with_val = create_validation_split_standalone(ds, val_ratio=args.val_ratio, seed=args.seed)
        except RuntimeError as e:
            print(f"❌ Failed to create validation split: {e}")
            sys.exit(1)

        print(f"\nSaving dataset with validation split...")
        try:
            save_dataset_with_validation_standalone(ds_with_val, args.dataset_dir)
        except RuntimeError as e:
            print(f"❌ Failed to save dataset: {e}")
            sys.exit(1)

        print(f"\n✓ Success! Dataset now has validation split:")
        print(f"  {', '.join(f'{k}={len(v)}' for k, v in ds_with_val.items())}")
        print(f"\nSaved to: {dataset_path}")
        print(f"\nVerify with: curl http://localhost:8000/api/parse/health | jq '.validation_size'")


if __name__ == "__main__":
    main()
