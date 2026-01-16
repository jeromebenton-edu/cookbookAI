# Data Quality: Teaching Moments in Historical Recipe Extraction

This document outlines data quality challenges encountered when extracting recipes from the 1896 *Boston Cooking-School Cook Book* and how we handle them for teaching purposes.

## Challenge 1: Recipe Variations Without Repeated Ingredients

**Problem:** Fanny Farmer frequently provides recipe variations that reference previous recipes rather than repeating all ingredients. For example:

> "For after-dinner coffee use twice the quantity of coffee, or half the amount of liquid, given in previous recipes."

This efficient writing style (which kept the book to 623 pages instead of 2000+) creates extraction challenges:
- The ML model doesn't have context from previous pages
- No discrete ingredient list appears on the page
- The text is primarily instructional/descriptive

**Solution:** Manual curation transforms imperfect ML output into usable recipe cards by:
1. Extracting the variation instruction as an "ingredient" (e.g., "Twice the quantity of coffee from previous recipes")
2. Breaking descriptive text into step-by-step instructions
3. Moving contextual information (historical notes, health warnings) to the notes section

**Example:** `after-dinner-coffee-p0074.json`
- **ML Output:** Empty ingredients, empty instructions (confidence: 0.0)
- **Curated Output:**
  - Ingredients: Reference to base recipe with modification
  - Instructions: 3 clear steps extracted from prose
  - Notes: Historical context (also called "Black Coffee or Café Noir") and health effects

## Challenge 2: Recipes with Missing Data

**Count:** 13 out of 391 recipes (~3%) have completely empty ingredients and instructions

**Categories:**
- Recipe variations (like After-Dinner Coffee above)
- Chapter headers (CHAPTER XI, CHAPTER XII, etc.)
- Incomplete OCR extraction (partial page scans)
- Non-recipe content misclassified as recipes

**Handling Strategy:**
1. **Chapter headers** → Keep for navigation but mark as non-recipe content
2. **Recipe variations** → Manually curate from scan images
3. **Incomplete extractions** → Fix individually or mark as "incomplete data"
4. **Non-recipes** → Reclassify or remove

## Teaching Value

This data quality issue demonstrates:

### 1. **Real-World ML Challenges**
- Models trained on modern recipe formats struggle with historical cookbook conventions
- Context dependency (cross-page references) breaks token-level classification
- OCR errors compound with classification errors

### 2. **Human-in-the-Loop Necessity**
- High-value applications require human curation
- ML provides 97% automation; humans fix the remaining 3%
- Domain expertise (understanding historical cookbook structure) is essential

### 3. **Crafting Usable Cards from Imperfect Data**
- Transform prose into structured data (ingredients, steps, notes)
- Preserve historical context while making content accessible
- Balance authenticity with usability

## Implementation Notes

**Files to Review:**
- `/frontend/public/recipes/boston/after-dinner-coffee-p0074.json` - Example of manual curation
- `/scripts/categorize_recipes.py` - Automated categorization despite missing ingredients
- `RecipeCard` component - UI gracefully handles missing fields

**Confidence Scores:**
The `field_confidence` metadata preserves ML output quality metrics:
```json
"field_confidence": {
  "title": 0.77,
  "ingredients": 0.0,  // Flags extraction failure
  "instructions": 0.0
}
```

This transparency allows users to understand data provenance and quality.

## Challenge 3: OCR Quality - The Achilles Heel

**Problem:** Tesseract OCR struggles with 1896 typography and two-column layout, creating garbled text that compounds downstream ML errors.

**Example:** Asparagus Soup (page 144)

*Should be:*
```
3 cups White Stock II. or III.
1 can asparagus
2 cups cold water
1 slice onion
¼ cup butter
¼ cup flour
2 cups scalded milk
Salt and pepper
```

*OCR output:*
```
8 2 1 Lean cups cups slice asparagus, cold onion. White
water. Stock Hi. or 111. 2 Salt 44 44 cups cup cup and
scalded butter. flour. pepper. milk. ¢
```

**Root Causes:**
1. **Multi-column layout** - Reading order jumps between columns
2. **Historical typography** - 19th century fonts misrecognized
3. **Special characters** - Fractions (¼, ½) become numbers or symbols
4. **No post-processing** - Raw Tesseract output without correction

**Why This Matters:**

OCR errors cascade through the pipeline:
- Bad OCR → Bad token classification → Garbled ingredients
- Even with perfect ML model, output quality limited by input quality
- **Garbage in, garbage out** principle in action

**Why LayoutLMv3 Still Works:**

Despite terrible OCR, the model can still:
- ✅ Identify recipe structure (title, ingredients, instructions sections)
- ✅ Extract bounding boxes for visual overlays
- ✅ Classify token roles even when text is wrong
- ✅ Provide confidence scores flagging quality issues

This demonstrates **robustness** - the model degrades gracefully with bad input.

**Solutions for Production Systems:**

1. **Modern Vision-Language Models**
   - GPT-4 Vision / Claude 3.5 Sonnet: Direct image → structured data
   - No OCR step needed - models "read" images directly
   - Cost: ~$0.01-0.10 per page (vs free Tesseract)

2. **Human-in-the-Loop Verification**
   - Flag low-confidence extractions (< 0.5) for manual review
   - Crowd-sourced correction (Amazon Mechanical Turk, Prolific)
   - Active learning: retrain on corrected examples

3. **Better OCR Engines**
   - Google Cloud Vision API (~$1.50/1000 pages)
   - Azure Computer Vision
   - Specialized historical document OCR tools

4. **Post-OCR Correction**
   - Language models to fix obvious errors
   - Dictionary lookup for ingredient terms
   - Pattern matching (e.g., "111" → "III", "44" → "¼")

**Teaching Value:**

This challenge demonstrates:
- **System thinking** - Understand entire pipeline, not just ML model
- **Pragmatic tradeoffs** - Free OCR vs paid APIs vs manual curation
- **Design for degradation** - Systems that work despite imperfect inputs
- **Honest assessment** - Acknowledging limitations builds trust

## Future Work

Potential improvements:
1. **Cross-page context** - Train models on multi-page sequences
2. **Recipe relationship detection** - Identify variation patterns automatically
3. **Active learning** - Prioritize manual review of low-confidence extractions
4. **Structured variation format** - Special UI for "doubles X from recipe Y" patterns
5. **OCR upgrade** - Evaluate GPT-4 Vision for high-quality re-extraction
6. **Crowdsourced corrections** - Community-driven data quality improvements

---

*This teaching portfolio demonstrates how AI-assisted digitization combines ML automation with human expertise to handle real-world complexity. By transparently showing both successes and limitations, it provides a realistic view of production ML systems.*
