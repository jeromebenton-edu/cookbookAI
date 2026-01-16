# Quick Start: Manual Labeling

## Ready to Label!

I've created a fast text-based labeling tool. Here's how to use it:

## Start Labeling Batch 1 (20 pages)

```bash
python manual_labeler.py data/label_studio/batch1_pages.txt
```

Or test with just the 7 demo pages first:
```bash
python manual_labeler.py 69,76,78,88,90,92,94
```

## How It Works

For each page, you'll see:
1. **Word list** with indices (first 50 words)
2. **Prompt** asking for the recipe title

### Label Recipe Title

Just type the recipe title words:
```
Page 69 RECIPE_TITLE: How to make Tea
```

Or type the word indices:
```
Page 69 RECIPE_TITLE: 5 6 7 8
```

### Optional: Label Section Headers

```
Page 76 RECIPE_TITLE: section: COCOA AND CHOCOLATE
Page 76 RECIPE_TITLE: Breakfast Cocoa
```

### Optional: Label Page Headers

```
Page 69 RECIPE_TITLE: header: BOSTON COOKING-SCHOOL COOK BOOK
Page 69 RECIPE_TITLE: How to make Tea
```

### Skip or Quit

- Type `skip` to skip a page
- Type `quit` to save and exit early

## Example Session

```
================================================================================
PAGE 69
================================================================================

Words on this page (first 50):
--------------------------------------------------------------------------------
  0: 38
  1: BOSTON
  2: COOKING-SCHOOL
  3: COOK
  4: BOOK.
  5: How
  6: to
  7: make
  8: Tea.
  9: 8
 10: teaspoons
 11: tea.
...
--------------------------------------------------------------------------------

Labeling Instructions:
  - Type the RECIPE_TITLE words (e.g., 'How to make Tea')
  - Or type word indices (e.g., '5 6 7 8')
  - Type 'section: WORD1 WORD2' for SECTION_HEADER
  - Type 'header: WORD1 WORD2' for PAGE_HEADER
  - Type 'skip' to skip this page
  - Type 'quit' to save and exit

Page 69 RECIPE_TITLE: How to make Tea
✓ Marked words [5, 6, 7, 8] as RECIPE_TITLE: 'How to make Tea.'

✓ Page 69 labeled (1/7)
```

## After Labeling

The tool will automatically:
1. Save labeled pages to `data/processed/v3_headers_titles/boston_v3_manual_labels.jsonl`
2. Show you the next steps for training

## Tips

- **Focus on RECIPE_TITLE first** - that's the priority
- **Don't overthink it** - if unclear, skip it
- **You can quit anytime** - progress is saved
- **2-3 minutes per page** - it's fast!

## Time Estimate

- **7 demo pages**: 15-20 minutes
- **20 Batch 1 pages**: 40-60 minutes

## Ready?

Start with the demo pages to get familiar:
```bash
python manual_labeler.py 69,76,78,88,90,92,94
```

Then move to Batch 1 when ready:
```bash
python manual_labeler.py data/label_studio/batch1_pages.txt
```

Good luck! 🚀
