#!/usr/bin/env python3
"""Prepare demo pages for Label Studio annotation."""

import json
from pathlib import Path
from datasets import load_from_disk

# Load dataset
ds_dict = load_from_disk("data/datasets/boston_layoutlmv3_v3/dataset_dict")
demo_ds = ds_dict["demo_eval"]

# Create Label Studio tasks
tasks = []
project_root = Path.cwd()

for idx, example in enumerate(demo_ds):
    image_path = Path(example["image_path"])

    # Convert to absolute path for Label Studio
    abs_image_path = (project_root / image_path).absolute()

    task = {
        "id": idx,
        "data": {
            "image": f"/data/local-files/?d={abs_image_path}",
            "page_num": int(image_path.stem),  # Extract page number from filename
        },
        "meta": {
            "words": example["words"],
            "bboxes": example["bboxes"],
            "image_width": example.get("width", 0),
            "image_height": example.get("height", 0),
        }
    }
    tasks.append(task)

# Save tasks
output_path = Path("data/label_studio/demo_pages_tasks.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Created {len(tasks)} Label Studio tasks")
print(f"Saved to: {output_path}")
print("\nPages to annotate:")
for task in tasks:
    print(f"  - Page {task['data']['page_num']:04d}: {len(task['meta']['words'])} words")
