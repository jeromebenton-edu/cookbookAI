#!/usr/bin/env python3
"""
Heuristic relabeling script to suggest PAGE_HEADER / SECTION_HEADER / RECIPE_TITLE labels.

This script takes existing v2 annotations (with generic TITLE) and applies heuristics
to suggest refined labels:
- PAGE_HEADER: Book titles, running headers, page numbers
- SECTION_HEADER: Category headings (BISCUITS, SOUPS, etc.)
- RECIPE_TITLE: Actual recipe titles

The suggestions are saved for human review and correction in Label Studio or similar.

Usage:
    python tools/relabel_suggest.py \\
        --input data/raw/weak_labeled.jsonl \\
        --output data/processed/v3_suggested.jsonl \\
        --stats relabel_stats.json

Input format: JSONL with fields:
    - page_num: int
    - words: List[str]
    - bboxes: List[List[int]] (normalized 0-1000)
    - labels: List[str] (v2 labels)
    - width: int
    - height: int

Output: Same format with updated labels and metadata field "label_source": "heuristic_v3"
"""

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RelabelSuggester:
    """Apply heuristics to suggest v3 labels from v2 annotations."""

    def __init__(self):
        # Common page header phrases
        self.page_header_phrases = {
            "BOSTON", "COOKING", "SCHOOL", "COOK", "BOOK",
            "RECIPES", "INDEX", "CONTENTS", "TABLE",
        }

        # Common section headers (all-caps category names)
        self.section_keywords = {
            "BISCUITS", "BREAKFAST", "CAKES", "BREADS",
            "SOUPS", "MEATS", "VEGETABLES", "DESSERTS",
            "SAUCES", "SALADS", "FISH", "POULTRY",
            "EGGS", "PIES", "PUDDINGS", "BEVERAGES",
        }

        # Track header patterns across pages
        self.header_patterns = defaultdict(list)  # {text: [page_nums]}

    def analyze_page_headers(self, records: List[Dict]) -> None:
        """
        Analyze all pages to identify repeated header patterns.

        Headers typically appear at consistent Y positions across pages.
        """
        logger.info("Analyzing header patterns across all pages...")

        for rec in records:
            page_num = rec.get("page_num", -1)
            words = rec.get("words", [])
            bboxes = rec.get("bboxes", [])
            height = rec.get("height", 1000)

            if not words or not bboxes:
                continue

            # Check top 15% of page for potential headers
            header_zone_y = height * 0.15

            for word, bbox in zip(words, bboxes):
                if len(bbox) != 4:
                    continue

                y_center = (bbox[1] + bbox[3]) / 2

                if y_center < header_zone_y:
                    # Track this word as potential header
                    self.header_patterns[word.upper()].append(page_num)

        # Find words that appear in top zone of many pages (likely headers)
        self.common_headers = {
            word for word, pages in self.header_patterns.items()
            if len(pages) >= 5  # Appears on 5+ pages
        }

        logger.info(f"Found {len(self.common_headers)} common header words")

    def classify_title_token(
        self,
        word: str,
        bbox: List[int],
        label: str,
        page_num: int,
        all_words: List[str],
        all_bboxes: List[List[int]],
        all_labels: List[str],
        height: int,
    ) -> str:
        """
        Classify a TITLE token into PAGE_HEADER / SECTION_HEADER / RECIPE_TITLE.

        Args:
            word: Token text
            bbox: [x1, y1, x2, y2] in image coordinates
            label: Current label (should be TITLE in v2)
            page_num: Page number
            all_words: All words on the page
            all_bboxes: All bboxes on the page
            all_labels: All labels on the page
            height: Image height

        Returns:
            Refined label: PAGE_HEADER, SECTION_HEADER, or RECIPE_TITLE
        """
        if label != "TITLE":
            # Not a title, keep as-is
            return label

        y_center = (bbox[1] + bbox[3]) / 2
        y_frac = y_center / height if height > 0 else 0

        word_upper = word.upper()

        # Rule 1: PAGE_HEADER heuristics
        # - Top 10% of page
        # - Matches common header phrases
        # - Contains page numbers
        if y_frac < 0.10:
            # Very top of page -> likely page header
            return "PAGE_HEADER"

        if word_upper in self.common_headers:
            # Repeated across many pages -> header
            return "PAGE_HEADER"

        if any(phrase in word_upper for phrase in self.page_header_phrases):
            # Matches book title phrases
            return "PAGE_HEADER"

        if re.match(r"^\d+$", word):
            # Just a number (page number)
            return "PAGE_HEADER"

        # Rule 2: SECTION_HEADER heuristics
        # - All-caps, short phrase
        # - Matches category keywords
        # - In upper 25% of page but not top 10%
        # - Above ingredient clusters
        if 0.10 < y_frac < 0.25:
            if word_upper == word and len(word) > 3:
                # All-caps word
                if any(kw in word_upper for kw in self.section_keywords):
                    return "SECTION_HEADER"

                # Check if it's a standalone all-caps line (likely section header)
                if self._is_standalone_caps_line(word, bbox, all_words, all_bboxes):
                    return "SECTION_HEADER"

        # Rule 3: RECIPE_TITLE (default for remaining TITLE labels)
        # - Mixed case or title case
        # - Near ingredient clusters (y_frac 0.15-0.50)
        # - Not in extreme top or bottom
        if 0.15 < y_frac < 0.50:
            # Check if followed by ingredients
            if self._is_near_ingredients(bbox, all_labels, all_bboxes):
                return "RECIPE_TITLE"

        # Default: If it has mixed case and isn't clearly a header, assume recipe title
        has_lowercase = any(c.islower() for c in word)
        if has_lowercase:
            return "RECIPE_TITLE"

        # Fallback: If all caps in middle of page, could be section header
        if y_frac < 0.40:
            return "SECTION_HEADER"

        return "RECIPE_TITLE"

    def _is_standalone_caps_line(
        self, word: str, bbox: List[int], all_words: List[str], all_bboxes: List[List[int]]
    ) -> bool:
        """Check if this is a standalone all-caps line (section header indicator)."""
        if word.upper() != word:
            return False

        # Find other words on same line (similar Y position)
        y_center = (bbox[1] + bbox[3]) / 2
        Y_TOLERANCE = 20  # pixels

        line_words = []
        for w, b in zip(all_words, all_bboxes):
            if len(b) == 4:
                w_y_center = (b[1] + b[3]) / 2
                if abs(w_y_center - y_center) < Y_TOLERANCE:
                    line_words.append(w)

        # If entire line is caps, likely a section header
        return all(w.upper() == w for w in line_words if len(w) > 1)

    def _is_near_ingredients(
        self, title_bbox: List[int], all_labels: List[str], all_bboxes: List[List[int]]
    ) -> bool:
        """Check if there are INGREDIENT_LINE labels near this title."""
        title_y_bottom = title_bbox[3]

        for label, bbox in zip(all_labels, all_bboxes):
            if label == "INGREDIENT_LINE" and len(bbox) == 4:
                ingredient_y_top = bbox[1]
                # If ingredient starts within 150 pixels below title, consider it "near"
                if 0 < (ingredient_y_top - title_y_bottom) < 150:
                    return True

        return False

    def relabel_page(self, record: Dict) -> Tuple[Dict, Dict]:
        """
        Apply heuristic relabeling to a single page.

        Args:
            record: Page record with v2 labels

        Returns:
            (updated_record, stats) where stats tracks label changes
        """
        page_num = record.get("page_num", -1)
        words = record.get("words", [])
        bboxes = record.get("bboxes", [])
        labels = record.get("labels", [])
        height = record.get("height", 1000)

        if not words or not labels or len(words) != len(labels):
            return record, {}

        new_labels = []
        changes = Counter()

        for i, (word, bbox, label) in enumerate(zip(words, bboxes, labels)):
            if label == "TITLE":
                new_label = self.classify_title_token(
                    word, bbox, label, page_num, words, bboxes, labels, height
                )
                if new_label != label:
                    changes[f"TITLE->{new_label}"] += 1
            else:
                new_label = label

            new_labels.append(new_label)

        # Update record
        updated_record = record.copy()
        updated_record["labels"] = new_labels
        updated_record["label_source"] = "heuristic_v3"
        updated_record["relabel_version"] = "v3_header_aware"

        return updated_record, dict(changes)

    def relabel_dataset(self, input_path: Path, output_path: Path) -> Dict:
        """
        Relabel entire dataset and save suggestions.

        Args:
            input_path: Input JSONL with v2 labels
            output_path: Output JSONL with v3 label suggestions

        Returns:
            Summary statistics
        """
        logger.info(f"Loading dataset from {input_path}...")
        records = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        logger.info(f"Loaded {len(records)} pages")

        # Analyze header patterns across all pages
        self.analyze_page_headers(records)

        # Relabel each page
        logger.info("Applying heuristic relabeling...")
        updated_records = []
        total_changes = Counter()

        for rec in records:
            updated_rec, changes = self.relabel_page(rec)
            updated_records.append(updated_rec)
            total_changes.update(changes)

        # Save updated records
        logger.info(f"Writing relabeled data to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in updated_records:
                f.write(json.dumps(rec) + "\n")

        # Compute summary stats
        stats = {
            "total_pages": len(records),
            "pages_processed": len(updated_records),
            "label_changes": dict(total_changes),
            "total_title_tokens_relabeled": sum(total_changes.values()),
        }

        # Count label distribution in output
        label_dist = Counter()
        for rec in updated_records:
            for lbl in rec.get("labels", []):
                label_dist[lbl] += 1

        stats["output_label_distribution"] = dict(label_dist)

        logger.info(f"✓ Relabeling complete!")
        logger.info(f"  Total TITLE tokens relabeled: {stats['total_title_tokens_relabeled']}")
        logger.info(f"  Label changes: {stats['label_changes']}")

        return stats


def main():
    parser = argparse.ArgumentParser(description="Suggest v3 labels from v2 annotations")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL (v2 labels)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL (v3 suggestions)")
    parser.add_argument("--stats", type=Path, help="Output stats JSON")

    args = parser.parse_args()

    suggester = RelabelSuggester()
    stats = suggester.relabel_dataset(args.input, args.output)

    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved stats to {args.stats}")

    # Print summary
    print("\n" + "=" * 80)
    print("RELABELING SUMMARY")
    print("=" * 80)
    print(f"Total pages processed: {stats['total_pages']}")
    print(f"Total TITLE tokens relabeled: {stats['total_title_tokens_relabeled']}")
    print("\nLabel changes:")
    for change, count in stats['label_changes'].items():
        print(f"  {change}: {count}")
    print("\nOutput label distribution (top 10):")
    sorted_labels = sorted(
        stats['output_label_distribution'].items(), key=lambda x: -x[1]
    )[:10]
    for label, count in sorted_labels:
        print(f"  {label}: {count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
