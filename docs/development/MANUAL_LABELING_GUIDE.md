# Manual Labeling Guide - Recipe Title Detection

## Summary

We need to manually label 20-40 pages with high-quality labels for:
- **RECIPE_TITLE**: Recipe names (e.g., "How to make Tea.", "Breakfast Cocoa.")
- **SECTION_HEADER**: Chapter/section headers (e.g., "BREAD AND BREAD MAKING")
- **PAGE_HEADER**: Page headers (e.g., "BOSTON COOKING-SCHOOL COOK BOOK")

## Why Manual Labeling?

The improved heuristics achieved 81% coverage but only 29% accuracy on verified pages. Training on noisy labels didn't improve model recall (stayed at 19.5%).

**Clean manual labels will give the model much better examples to learn from.**

## Recommended Batches

### Batch 1: Pages with Section Headers (20 pages)
**Pages**: 2, 17, 24, 30, 56, 66, 88, 100, 178, 196, 224, 248, 260, 272, 434, 466, 490, 502, 522, 534

**Why these?**
- Have clear section headers detected by heuristics
- Easier to label (structure is obvious)
- Good for training section header → recipe title pattern

**Time estimate**: 40-60 minutes (2-3 min/page)

### Batch 3: Diverse Coverage (10 pages)
**Pages**: 5, 10, 15, 20, 25, 300, 320, 340, 550, 580

**Why these?**
- Cover early (intro), middle, and late (index) sections
- Different types of content (not just recipes)
- Help model learn what's NOT a recipe title

**Time estimate**: 20-30 minutes (2-3 min/page)

## Labeling Instructions

### What to Label

1. **RECIPE_TITLE** (Priority!)
   - The name of a recipe at the start of a recipe
   - Examples:
     - "How to make Tea."
     - "Breakfast Cocoa."
     - "Graham Bread."
     - "Indian Bread"
   - Characteristics:
     - 2-5 words typically
     - Appears before ingredients
     - Centered or left-aligned
     - First letter capitalized
     - Often ends with period

2. **SECTION_HEADER**
   - Chapter or section headers
   - Examples:
     - "BREAD AND BREAD MAKING"
     - "COCOA AND CHOCOLATE"
     - "FRUIT DRINKS"
   - Characteristics:
     - ALL CAPS
     - At very top of page (top 5%)
     - Contains generic words (AND, MAKING, THE, etc.)
     - Centered

3. **PAGE_HEADER**
   - Running headers at top of every page
   - Examples:
     - "BOSTON COOKING-SCHOOL COOK BOOK"
     - Page numbers
   - Characteristics:
     - At very top (top 3%)
     - Same on multiple pages
     - Not recipe-specific content

### What NOT to Label (Leave as O)

- Ingredient lists (already labeled as INGREDIENT_LINE)
- Instructions (already labeled as INSTRUCTION_STEP)
- Body text, explanations
- Random phrases that happen to contain food words

## Two Options for Labeling

### Option 1: Label Studio (Visual, Slower)

**Pros**:
- See the actual page images
- Draw bounding boxes around text
- More intuitive

**Cons**:
- Slower (need to draw boxes)
- Setup overhead

**Steps**:
1. Restart Label Studio:
   ```bash
   pkill -f label-studio
   ./start_label_studio.sh
   ```

2. Create new project for Batch 1
3. Import 20 pages
4. Label and export

### Option 2: Text-Based Labeling (Fast, Recommended)

**Pros**:
- Much faster (just type words)
- No server needed
- Can use regex/search

**Cons**:
- Don't see page images
- Need to identify titles from word lists

**Steps**:
I'll create a simple Python script where you:
1. See word list for each page
2. Type the recipe title words
3. Script automatically updates labels
4. Export to JSONL

## Recommended Workflow

**Start with Batch 1 (20 pages)**:

1. **Label the 20 pages** (40-60 minutes)
   - Focus on RECIPE_TITLE first
   - Add SECTION_HEADER and PAGE_HEADER if obvious
   - Don't overthink it - just mark the clear recipe names

2. **Rebuild dataset** with manual labels
   ```bash
   python tools/build_v3_dataset.py \
     --input data/processed/v3_headers_titles/boston_v3_manual_batch1.jsonl \
     --output data/datasets/boston_layoutlmv3_v3_manual_batch1
   ```

3. **Train model** (30 minutes)
   ```bash
   python tools/train_v3_model.py \
     --dataset data/datasets/boston_layoutlmv3_v3_manual_batch1/dataset_dict \
     --output models/layoutlmv3_v3_manual_batch1 \
     --epochs 20
   ```

4. **Evaluate on demo pages**
   - Check if RECIPE_TITLE recall improved
   - If yes → continue with more labeling
   - If no → analyze what went wrong

5. **Iterate**: Label more pages, retrain, evaluate

## Expected Improvement

With 20 clean manual labels:
- **Current recall**: 19.5%
- **Target recall**: 40-60% (2-3x improvement)
- **With 40 pages**: 60-80% recall possible

## Quality Over Quantity

**20 perfectly labeled pages > 100 noisy heuristic pages**

Focus on accuracy:
- Only label what you're confident about
- Skip ambiguous cases
- Consistency is key

## Next Steps

Ready to start? Let me know if you want:
1. **Option 1**: Set up Label Studio for visual labeling
2. **Option 2**: Create fast text-based labeling script
3. **Something else**: Different approach?

I recommend Option 2 (text-based) for speed and efficiency.
