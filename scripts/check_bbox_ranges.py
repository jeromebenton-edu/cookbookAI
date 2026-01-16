#!/usr/bin/env python3
"""Check bbox coordinate ranges in dataset."""

from pathlib import Path
from datasets import load_from_disk

# Load dataset
dataset_path = Path("data/datasets/boston_layoutlmv3_v3/dataset_dict")
print(f"Loading dataset from {dataset_path}...")
ds_dict = load_from_disk(dataset_path)

print("\nChecking bbox ranges across all splits...")

for split_name, ds in ds_dict.items():
    print(f"\n{split_name.upper()} ({len(ds)} examples):")

    max_coords = []
    min_coords = []
    out_of_range_examples = []

    for i, example in enumerate(ds):
        bboxes = example["bboxes"]
        if not bboxes:
            continue

        flat_coords = [coord for bbox in bboxes for coord in bbox]
        if not flat_coords:
            continue

        max_coord = max(flat_coords)
        min_coord = min(flat_coords)

        max_coords.append(max_coord)
        min_coords.append(min_coord)

        if max_coord > 1000 or min_coord < 0:
            out_of_range_examples.append({
                'index': i,
                'image': example['image_path'],
                'min': min_coord,
                'max': max_coord,
            })

    if max_coords:
        print(f"  Overall range: [{min(min_coords)}, {max(max_coords)}]")
        print(f"  Examples with coords > 1000: {sum(1 for m in max_coords if m > 1000)}")
        print(f"  Examples with coords < 0: {sum(1 for m in min_coords if m < 0)}")

        if out_of_range_examples:
            print(f"\n  OUT OF RANGE EXAMPLES ({len(out_of_range_examples)}):")
            for ex in out_of_range_examples[:10]:  # Show first 10
                print(f"    Index {ex['index']}: {ex['image']} - range [{ex['min']}, {ex['max']}]")
            if len(out_of_range_examples) > 10:
                print(f"    ... and {len(out_of_range_examples) - 10} more")

print("\n" + "=" * 80)
print("DONE")
