# Demo Examples

This directory contains bundled demo examples for the `/demo` route. Each example is a self-consistent package containing:

- **page.png** - The actual scanned cookbook page
- **prediction.json** - The ML extraction output from this exact page
- **meta.json** - Metadata and validation fields

## Structure

```
demo_examples/
├── example_01/
│   ├── page.png         # Scanned page from page 50 (Boston Cream Pie)
│   ├── prediction.json  # Extraction output
│   └── meta.json        # Metadata with validation
└── example_02/
    ├── page.png         # Scanned page from page 75 (Chocolate Fudge)
    ├── prediction.json  # Extraction output
    └── meta.json        # Metadata with validation
```

## Validation

Each example includes validation fields in `meta.json` to ensure the scan and prediction match:

```json
{
  "title": "Boston Cream Pie",
  "difficulty": "medium",
  "page_num": 50,
  "expectedTitle": "Boston Cream Pie",
  "expectedKeywords": ["boston", "cream", "pie"]
}
```

### Validation Rules

The loader validates that:
1. **Title match**: Prediction title contains the expectedTitle (case-insensitive)
2. **Keyword match**: Prediction title contains at least one expectedKeyword
3. **Page number**: Prediction page_num matches meta page_num

If validation fails:
- **Development**: Console warning with details
- **Production**: Banner shown in UI: "Demo assets misconfigured"

## Adding New Examples

1. Create a new directory: `demo_examples/example_03/`

2. Add the scanned page image:
   ```bash
   cp /path/to/cookbook/page.png example_03/page.png
   ```

3. Create `prediction.json` with the ML extraction output for that page

4. Create `meta.json`:
   ```json
   {
     "title": "Recipe Title",
     "difficulty": "easy|medium|hard",
     "description": "Brief description",
     "tags": ["category", "type"],
     "page_num": 123,
     "expectedTitle": "Recipe Title",
     "expectedKeywords": ["keyword1", "keyword2"]
   }
   ```

5. Update `lib/bundledExamples.ts`:
   - Import the new example data
   - Add to `DEMO_EXAMPLES_RAW` array

6. Copy image to public directory:
   ```bash
   mkdir -p public/demo_examples/example_03
   cp src/demo_examples/example_03/page.png public/demo_examples/example_03/
   ```

## Important Rules

**The scan image and prediction MUST match:**
- The scan must be the actual source page for the prediction JSON
- The prediction title must appear on the scan
- Page numbers must align

This ensures demo trust - users see the real page the AI extracted from, not a random unrelated scan.
