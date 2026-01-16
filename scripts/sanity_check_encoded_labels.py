#!/usr/bin/env python3
"""
Sanity check for encoded training labels.

Validates that the encoding pipeline produces usable supervision signal:
- Labels are not all masked (-100)
- Non-O labels exist in reasonable proportion
- Label distribution matches expectations
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_from_disk, DatasetDict
from transformers import LayoutLMv3Processor

# Add project root to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.modeling.alignment import encode_example


def analyze_encoded_example(
    example: dict,
    processor: LayoutLMv3Processor,
    label_map: Dict[int, str],
    max_length: int = 512,
) -> Dict:
    """Encode an example and analyze its labels."""
    # Encode using the same function as training
    encoded = encode_example(example, processor, label_pad_token_id=-100, max_length=max_length)

    labels = encoded['labels']
    input_ids = encoded['input_ids']

    # Count labels
    label_counts = Counter(labels)
    total_tokens = len(labels)
    masked_tokens = label_counts.get(-100, 0)
    unmasked_tokens = total_tokens - masked_tokens

    # Count by label name (excluding masked)
    label_name_counts = Counter()
    for label_id in labels:
        if label_id != -100:
            label_name = label_map.get(label_id, f"UNKNOWN_{label_id}")
            label_name_counts[label_name] += 1

    # Find non-O tokens
    non_o_count = sum(count for label, count in label_name_counts.items() if label != 'O')
    non_o_ratio = non_o_count / unmasked_tokens if unmasked_tokens > 0 else 0.0

    # Collect sample labeled tokens (first 5 non-O or interesting ones)
    sample_tokens = []
    for i, (token_id, label_id) in enumerate(zip(input_ids, labels)):
        if label_id != -100 and len(sample_tokens) < 5:
            label_name = label_map.get(label_id, f"UNKNOWN_{label_id}")
            if label_name != 'O':  # Prefer non-O examples
                token_text = processor.tokenizer.decode([token_id])
                sample_tokens.append((i, token_text, label_name))

    return {
        'total_tokens': total_tokens,
        'masked_tokens': masked_tokens,
        'unmasked_tokens': unmasked_tokens,
        'masked_ratio': masked_tokens / total_tokens if total_tokens > 0 else 0.0,
        'label_counts': dict(label_counts),
        'label_name_counts': dict(label_name_counts),
        'non_o_count': non_o_count,
        'non_o_ratio': non_o_ratio,
        'sample_tokens': sample_tokens,
        'has_ingredient': any('INGREDIENT' in label for label in label_name_counts.keys()),
        'has_instruction': any('INSTRUCTION' in label for label in label_name_counts.keys()),
        'has_title': 'TITLE' in label_name_counts,
    }


def print_example_analysis(idx: int, analysis: Dict, verbose: bool = False):
    """Print analysis for a single example."""
    print(f"\n{'='*80}")
    print(f"Example {idx}")
    print(f"{'='*80}")
    print(f"Total tokens:     {analysis['total_tokens']}")
    print(f"Masked tokens:    {analysis['masked_tokens']} ({analysis['masked_ratio']:.1%})")
    print(f"Unmasked tokens:  {analysis['unmasked_tokens']}")
    print(f"Non-O tokens:     {analysis['non_o_count']} ({analysis['non_o_ratio']:.1%} of unmasked)")

    print(f"\nLabel distribution (unmasked):")
    for label_name, count in sorted(analysis['label_name_counts'].items(), key=lambda x: -x[1]):
        ratio = count / analysis['unmasked_tokens'] if analysis['unmasked_tokens'] > 0 else 0
        print(f"  {label_name:20s}: {count:5d} ({ratio:6.1%})")

    if verbose and analysis['sample_tokens']:
        print(f"\nSample labeled tokens:")
        for pos, token, label in analysis['sample_tokens']:
            print(f"  [{pos:3d}] {token:20s} -> {label}")


def aggregate_statistics(analyses: List[Dict]) -> Dict:
    """Aggregate statistics across all analyzed examples."""
    total_examples = len(analyses)

    # Aggregate counts
    global_label_counts = Counter()
    global_masked = 0
    global_total = 0
    examples_with_ingredient = 0
    examples_with_instruction = 0
    examples_with_title = 0
    examples_with_any_non_o = 0

    for analysis in analyses:
        global_total += analysis['total_tokens']
        global_masked += analysis['masked_tokens']

        for label_name, count in analysis['label_name_counts'].items():
            global_label_counts[label_name] += count

        if analysis['has_ingredient']:
            examples_with_ingredient += 1
        if analysis['has_instruction']:
            examples_with_instruction += 1
        if analysis['has_title']:
            examples_with_title += 1
        if analysis['non_o_count'] > 0:
            examples_with_any_non_o += 1

    global_unmasked = global_total - global_masked
    non_o_total = sum(count for label, count in global_label_counts.items() if label != 'O')

    return {
        'total_examples': total_examples,
        'global_total_tokens': global_total,
        'global_masked_tokens': global_masked,
        'global_unmasked_tokens': global_unmasked,
        'global_masked_ratio': global_masked / global_total if global_total > 0 else 0.0,
        'global_label_counts': dict(global_label_counts),
        'global_non_o_count': non_o_total,
        'global_non_o_ratio': non_o_total / global_unmasked if global_unmasked > 0 else 0.0,
        'examples_with_ingredient_pct': examples_with_ingredient / total_examples if total_examples > 0 else 0.0,
        'examples_with_instruction_pct': examples_with_instruction / total_examples if total_examples > 0 else 0.0,
        'examples_with_title_pct': examples_with_title / total_examples if total_examples > 0 else 0.0,
        'examples_with_any_non_o_pct': examples_with_any_non_o / total_examples if total_examples > 0 else 0.0,
    }


def print_aggregate_statistics(stats: Dict):
    """Print aggregate statistics."""
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")
    print(f"Total examples analyzed:  {stats['total_examples']}")
    print(f"Total tokens:             {stats['global_total_tokens']}")
    print(f"Masked tokens:            {stats['global_masked_tokens']} ({stats['global_masked_ratio']:.1%})")
    print(f"Unmasked tokens:          {stats['global_unmasked_tokens']}")
    print(f"Non-O tokens:             {stats['global_non_o_count']} ({stats['global_non_o_ratio']:.1%} of unmasked)")

    print(f"\nGlobal label distribution (unmasked tokens):")
    for label_name, count in sorted(stats['global_label_counts'].items(), key=lambda x: -x[1]):
        ratio = count / stats['global_unmasked_tokens'] if stats['global_unmasked_tokens'] > 0 else 0
        print(f"  {label_name:20s}: {count:6d} ({ratio:6.1%})")

    print(f"\nExamples with specific labels:")
    print(f"  INGREDIENT_LINE:  {stats['examples_with_ingredient_pct']:6.1%} ({int(stats['examples_with_ingredient_pct'] * stats['total_examples'])}/{stats['total_examples']})")
    print(f"  INSTRUCTION_STEP: {stats['examples_with_instruction_pct']:6.1%} ({int(stats['examples_with_instruction_pct'] * stats['total_examples'])}/{stats['total_examples']})")
    print(f"  TITLE:            {stats['examples_with_title_pct']:6.1%} ({int(stats['examples_with_title_pct'] * stats['total_examples'])}/{stats['total_examples']})")
    print(f"  Any non-O:        {stats['examples_with_any_non_o_pct']:6.1%} ({int(stats['examples_with_any_non_o_pct'] * stats['total_examples'])}/{stats['total_examples']})")


def check_failures(stats: Dict) -> Tuple[bool, List[str]]:
    """Check if statistics indicate failure conditions."""
    failures = []

    # Check 1: Too many masked tokens
    if stats['global_masked_ratio'] > 0.95:
        failures.append(
            f"❌ FAIL: {stats['global_masked_ratio']:.1%} of tokens are masked (>95% threshold). "
            f"Labels are being almost entirely masked out during encoding."
        )

    # Check 2: Too few non-O labels
    if stats['global_non_o_ratio'] < 0.01:
        failures.append(
            f"❌ FAIL: Only {stats['global_non_o_ratio']:.2%} of unmasked tokens have non-O labels (<1% threshold). "
            f"Insufficient supervision signal for training."
        )

    # Check 3: Too few examples with non-O labels
    if stats['examples_with_any_non_o_pct'] < 0.10:
        failures.append(
            f"❌ FAIL: Only {stats['examples_with_any_non_o_pct']:.1%} of examples contain any non-O labels (<10% threshold). "
            f"Most examples have no supervision signal."
        )

    # Check 4: No INGREDIENT or INSTRUCTION labels at all
    if stats['examples_with_ingredient_pct'] == 0.0:
        failures.append(
            f"⚠️  WARNING: No examples contain INGREDIENT_LINE labels. "
            f"Training will not learn to predict ingredients."
        )

    if stats['examples_with_instruction_pct'] == 0.0:
        failures.append(
            f"⚠️  WARNING: No examples contain INSTRUCTION_STEP labels. "
            f"Training will not learn to predict instructions."
        )

    return len(failures) > 0, failures


def main():
    parser = argparse.ArgumentParser(description="Sanity check encoded training labels")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory (should contain dataset_dict)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of examples to analyze per split (default: 20)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for encoding (default: 512)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-example analysis"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Which splits to analyze (default: train validation)"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"❌ ERROR: Dataset directory not found: {dataset_dir}")
        return 1

    # Load dataset
    print(f"Loading dataset from {dataset_dir}...")
    if (dataset_dir / "dataset_dict").exists():
        ds = load_from_disk(str(dataset_dir / "dataset_dict"))
    else:
        ds = load_from_disk(str(dataset_dir))

    if not isinstance(ds, DatasetDict):
        print(f"❌ ERROR: Expected DatasetDict, got {type(ds)}")
        return 1

    print(f"Available splits: {list(ds.keys())}")

    # Load label map
    label_map_path = dataset_dir / "label_map.json"
    if not label_map_path.exists():
        label_map_path = dataset_dir.parent / "label_map.json"

    if not label_map_path.exists():
        print(f"❌ ERROR: label_map.json not found near {dataset_dir}")
        return 1

    import json
    with open(label_map_path) as f:
        label_data = json.load(f)
        id2label = {int(k): v for k, v in label_data['id2label'].items()}

    print(f"Label map: {id2label}")

    # Load processor
    print("Loading LayoutLMv3 processor...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    # Analyze each split
    all_passed = True

    for split_name in args.splits:
        if split_name not in ds:
            print(f"\n⚠️  WARNING: Split '{split_name}' not found in dataset, skipping")
            continue

        split_data = ds[split_name]
        num_examples = min(args.num_samples, len(split_data))

        print(f"\n{'#'*80}")
        print(f"# Analyzing split: {split_name} ({num_examples} samples)")
        print(f"{'#'*80}")

        # Analyze samples
        analyses = []
        for i in range(num_examples):
            example = split_data[i]
            analysis = analyze_encoded_example(example, processor, id2label, args.max_length)
            analyses.append(analysis)

            if args.verbose:
                print_example_analysis(i, analysis, verbose=True)

        # Aggregate and print statistics
        stats = aggregate_statistics(analyses)
        print_aggregate_statistics(stats)

        # Check for failures
        has_failures, failure_msgs = check_failures(stats)

        if has_failures:
            print(f"\n{'!'*80}")
            print("SANITY CHECK FAILURES DETECTED")
            print(f"{'!'*80}")
            for msg in failure_msgs:
                print(f"\n{msg}")
            all_passed = False
        else:
            print(f"\n{'='*80}")
            print("✓ SANITY CHECK PASSED")
            print(f"{'='*80}")
            print("Labels look good! Encoding preserves usable supervision signal.")

    # Final result
    if not all_passed:
        print(f"\n{'!'*80}")
        print("❌ SANITY CHECK FAILED")
        print(f"{'!'*80}")
        print("\nLabels are not usable for training. Please investigate:")
        print("1. Check that dataset has correct 'labels' column with non-O annotations")
        print("2. Verify encode_example() is aligning labels correctly")
        print("3. Check for issues with tokenizer word boundaries")
        print("4. Review label masking logic (should only mask special tokens)")
        return 1

    print(f"\n{'='*80}")
    print("✓ ALL SANITY CHECKS PASSED")
    print(f"{'='*80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
