#!/bin/bash
# Re-parse all recipes with OCR post-processing improvements
#
# This script:
# 1. Backs up existing recipes
# 2. Re-parses all 413 recipes with OCR corrections applied
# 3. Compares results to show improvements

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
RECIPES_DIR="$PROJECT_ROOT/frontend/public/recipes/boston"
BACKUP_DIR="$PROJECT_ROOT/frontend/public/recipes/boston_backup_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "Re-parsing Recipes with OCR Improvements"
echo "========================================="
echo ""

# Create backup
echo "1. Creating backup of existing recipes..."
if [ -d "$RECIPES_DIR" ]; then
    cp -r "$RECIPES_DIR" "$BACKUP_DIR"
    echo "   ✓ Backup created at: $BACKUP_DIR"
else
    echo "   ! No existing recipes found"
fi
echo ""

# Count existing recipes with titles
echo "2. Analyzing existing recipes..."
EXISTING_TITLES=$(find "$RECIPES_DIR" -name "*.json" -exec jq -r '.title' {} \; 2>/dev/null | grep -v "Recipe from page" | wc -l || echo "0")
TOTAL_RECIPES=$(find "$RECIPES_DIR" -name "*.json" 2>/dev/null | wc -l || echo "0")
echo "   Current recipes: $TOTAL_RECIPES"
echo "   Recipes with actual titles: $EXISTING_TITLES"
echo ""

# Re-parse with OCR improvements
echo "3. Re-parsing all recipes with OCR post-processing..."
cd "$PROJECT_ROOT"
python tools/parse_full_cookbook.py

echo ""
echo "4. Analyzing new results..."
NEW_TITLES=$(find "$RECIPES_DIR" -name "*.json" -exec jq -r '.title' {} \; 2>/dev/null | grep -v "Recipe from page" | wc -l || echo "0")
NEW_TOTAL=$(find "$RECIPES_DIR" -name "*.json" 2>/dev/null | wc -l || echo "0")
echo "   New recipes: $NEW_TOTAL"
echo "   Recipes with actual titles: $NEW_TITLES"
echo ""

# Calculate improvement
TITLE_IMPROVEMENT=$((NEW_TITLES - EXISTING_TITLES))
echo "========================================="
echo "RESULTS"
echo "========================================="
echo "Total recipes: $NEW_TOTAL"
echo "Titles extracted: $NEW_TITLES (was $EXISTING_TITLES)"
if [ $TITLE_IMPROVEMENT -gt 0 ]; then
    echo "✓ Improvement: +$TITLE_IMPROVEMENT titles"
elif [ $TITLE_IMPROVEMENT -lt 0 ]; then
    echo "! Regression: $TITLE_IMPROVEMENT titles"
else
    echo "= No change in title count"
fi
echo ""
echo "Backup available at: $BACKUP_DIR"
echo "========================================="
