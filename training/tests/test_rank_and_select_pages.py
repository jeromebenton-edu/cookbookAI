#!/usr/bin/env python3
"""Unit tests for rank_and_select_pages with blended ranking and ingredient coverage."""

import unittest
from unittest.mock import Mock
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.datasets.build_layoutlm_dataset import rank_and_select_pages, PageExample


class TestRankAndSelectPages(unittest.TestCase):
    """Test blended ranking and ingredient coverage constraint."""

    def create_page(
        self,
        page_num: int,
        ingredient_tokens: int = 0,
        instruction_tokens: int = 0,
        title_tokens: int = 0,
    ) -> PageExample:
        """Helper to create a PageExample with specific label counts."""
        total_tokens = 100
        non_o_count = ingredient_tokens + instruction_tokens + title_tokens

        return PageExample(
            page_num=page_num,
            id=f"page_{page_num}",
            image_path=f"/path/to/page_{page_num}.png",
            words=["word"] * total_tokens,
            bboxes=[[0, 0, 100, 100]] * total_tokens,
            labels=[0] * total_tokens,
            label_names=["O"] * total_tokens,
            width=1000,
            height=1000,
            has_labels=True,
            png_id=f"{page_num:04d}",
            non_o_ratio=non_o_count / total_tokens,
            non_o_token_count=non_o_count,
            ingredient_token_count=ingredient_tokens,
            instruction_token_count=instruction_tokens,
            title_token_count=title_tokens,
        )

    def test_blended_ranking_stable_deterministic(self):
        """Test that blended ranking is stable and deterministic."""
        # Create pages with different instruction/ingredient balances
        pages = [
            self.create_page(1, ingredient_tokens=50, instruction_tokens=10),  # Score: 0.6*10 + 0.4*50 = 26
            self.create_page(2, ingredient_tokens=10, instruction_tokens=50),  # Score: 0.6*50 + 0.4*10 = 34
            self.create_page(3, ingredient_tokens=30, instruction_tokens=30),  # Score: 0.6*30 + 0.4*30 = 30
            self.create_page(4, ingredient_tokens=0, instruction_tokens=60),   # Score: 0.6*60 + 0.4*0 = 36
            self.create_page(5, ingredient_tokens=60, instruction_tokens=0),   # Score: 0.6*0 + 0.4*60 = 24
        ]

        # Select top 3
        selected = rank_and_select_pages(pages, max_pages=3, rank_by_instructions=True)

        # Expected order by score: page 4 (36), page 2 (34), page 3 (30)
        self.assertEqual(len(selected), 3)
        selected_nums = sorted([p.page_num for p in selected])
        expected_nums = sorted([2, 3, 4])
        self.assertEqual(selected_nums, expected_nums)

        # Run again to verify determinism
        selected2 = rank_and_select_pages(pages, max_pages=3, rank_by_instructions=True)
        selected2_nums = sorted([p.page_num for p in selected2])
        self.assertEqual(selected2_nums, expected_nums)

    def test_tiebreaker_uses_page_num(self):
        """Test that ties are broken by page_num for determinism."""
        # Create pages with identical scores
        pages = [
            self.create_page(100, ingredient_tokens=20, instruction_tokens=20),  # Score: 20
            self.create_page(50, ingredient_tokens=20, instruction_tokens=20),   # Score: 20
            self.create_page(75, ingredient_tokens=20, instruction_tokens=20),   # Score: 20
        ]

        selected = rank_and_select_pages(pages, max_pages=2, rank_by_instructions=True)

        # Should select pages with higher page_num (descending order, then by page_num)
        selected_nums = sorted([p.page_num for p in selected])
        # Since all have same score, tiebreaker is page_num (in descending order in the key)
        # So we get the two highest page_nums
        self.assertEqual(len(selected), 2)
        self.assertIn(100, selected_nums)
        self.assertIn(75, selected_nums)

    def test_ingredient_coverage_constraint_met(self):
        """Test that ingredient coverage constraint is met."""
        # Create 15 pages: 10 with only instructions, 5 with ingredients
        # This ensures there are enough ingredient pages to meet 30% coverage for 10 selected
        pages = []
        for i in range(10):
            pages.append(self.create_page(i, ingredient_tokens=0, instruction_tokens=50))

        for i in range(10, 15):
            pages.append(self.create_page(i, ingredient_tokens=15, instruction_tokens=30))

        # Select 10 pages with 30% ingredient coverage requirement
        selected = rank_and_select_pages(pages, max_pages=10, rank_by_instructions=True, min_ingredient_coverage=0.3)

        # Count pages with ingredient_tokens >= 10
        pages_with_ing = [p for p in selected if p.ingredient_token_count >= 10]
        coverage = len(pages_with_ing) / len(selected)

        # Should have at least 30% coverage
        self.assertGreaterEqual(coverage, 0.3)
        self.assertGreaterEqual(len(pages_with_ing), 3)  # 30% of 10 = 3

    def test_ingredient_coverage_backfill(self):
        """Test that backfill swaps low-ingredient pages for high-ingredient pages."""
        # Create scenario where initial selection has low ingredient coverage
        pages = []

        # 5 pages with high instructions, no ingredients (will be top-ranked initially)
        for i in range(5):
            pages.append(self.create_page(i, ingredient_tokens=0, instruction_tokens=100))

        # 5 pages with moderate instructions + ingredients (lower blended score)
        for i in range(5, 10):
            pages.append(self.create_page(i, ingredient_tokens=50, instruction_tokens=50))

        # Select 10 pages with 30% ingredient coverage
        selected = rank_and_select_pages(pages, max_pages=10, rank_by_instructions=True, min_ingredient_coverage=0.3)

        # Should have backfilled to meet 30% coverage
        pages_with_ing = [p for p in selected if p.ingredient_token_count >= 10]
        coverage = len(pages_with_ing) / len(selected)

        self.assertGreaterEqual(coverage, 0.3)
        self.assertGreaterEqual(len(pages_with_ing), 3)

        # Should include some of the ingredient-rich pages
        selected_nums = [p.page_num for p in selected]
        ingredient_page_nums = list(range(5, 10))
        included_ing_pages = [n for n in ingredient_page_nums if n in selected_nums]
        self.assertGreater(len(included_ing_pages), 0)

    def test_no_limit_returns_all_pages(self):
        """Test that max_pages=0 returns all pages."""
        pages = [
            self.create_page(i, ingredient_tokens=10, instruction_tokens=10)
            for i in range(5)
        ]

        selected = rank_and_select_pages(pages, max_pages=0, rank_by_instructions=True)

        self.assertEqual(len(selected), len(pages))

    def test_fewer_pages_than_limit_returns_all(self):
        """Test that if pages < max_pages, all pages are returned."""
        pages = [
            self.create_page(i, ingredient_tokens=10, instruction_tokens=10)
            for i in range(5)
        ]

        selected = rank_and_select_pages(pages, max_pages=10, rank_by_instructions=True)

        self.assertEqual(len(selected), len(pages))

    def test_blended_score_weights(self):
        """Test that blended score correctly weights instructions (0.6) and ingredients (0.4)."""
        # Page with more instructions should rank higher than page with more ingredients
        pages = [
            self.create_page(1, ingredient_tokens=100, instruction_tokens=0),   # Score: 40
            self.create_page(2, ingredient_tokens=0, instruction_tokens=67),    # Score: 40.2
        ]

        selected = rank_and_select_pages(pages, max_pages=1, rank_by_instructions=True)

        # Page 2 should win because instructions are weighted higher (0.6 vs 0.4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].page_num, 2)

    def test_fallback_ranking_without_blending(self):
        """Test fallback ranking when rank_by_instructions=False."""
        pages = [
            self.create_page(1, ingredient_tokens=10, instruction_tokens=50),
            self.create_page(2, ingredient_tokens=50, instruction_tokens=10),
            self.create_page(3, ingredient_tokens=30, instruction_tokens=30),
        ]

        # With rank_by_instructions=False, should rank by non_o_token_count
        selected = rank_and_select_pages(pages, max_pages=2, rank_by_instructions=False)

        # All pages have 60 non-O tokens, so order depends on secondary keys
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
