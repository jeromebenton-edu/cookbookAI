# Label Studio Setup Guide

## Server Status
✓ Label Studio is running at **http://localhost:8080**

## What to Label
Focus on labeling **RECIPE_TITLE** on these 7 demo pages:
- Page 0069: "How to make Tea."
- Page 0076: "Breakfast Cocoa"
- Page 0078: "Claret Punch"
- Page 0088-0094: Various bread recipes

## Setup Instructions

### 1. Open Label Studio
Open your browser to: **http://localhost:8080**

### 2. Create Account (First Time Only)
Create a user account - use any email/password you want (stored locally)

### 3. Create a New Project
- Click "Create Project"
- Name it: **CookbookAI Demo Pages**
- Description: "Labeling recipe titles and components on demo pages"

### 4. Configure Labeling Interface
In the project settings (or during creation):
1. Go to "Settings" → "Labeling Interface"
2. Click "Code" view
3. Copy and paste the contents of: **data/label_studio/config.xml**
4. Click "Save"

### 5. Import Tasks
1. Go to "Settings" → "Import"
2. Upload the file: **data/label_studio/demo_pages_tasks.json**
3. Click "Import"
4. You should see 7 tasks imported

### 6. Start Labeling!

#### Labeling Instructions:
1. Click on the first task to open the image
2. Use the rectangle tool to draw boxes around text regions
3. **Priority: Focus on RECIPE_TITLE**
   - Example: "How to make Tea." at the top of page 69
   - These are the recipe names at the start of each recipe
   - Typically larger text, positioned near the top before ingredients

4. You can also label other elements if you want:
   - **INGREDIENT_LINE**: Individual ingredient items
   - **INSTRUCTION_STEP**: Cooking instruction steps
   - **PAGE_HEADER**: Headers like "BOSTON COOKING-SCHOOL COOK BOOK"

#### Keyboard Shortcuts:
- `1` = RECIPE_TITLE (red)
- `2` = PAGE_HEADER (blue)
- `3` = SECTION_HEADER (purple)
- `4` = INGREDIENT_LINE (green)
- `5` = INSTRUCTION_STEP (orange)
- `Ctrl+Enter` = Submit task
- `Ctrl+Backspace` = Skip task

### 7. Export Annotations When Done
1. Go to project page
2. Click "Export"
3. Choose "JSON" format
4. Save to: **data/label_studio/demo_annotations.json**

## Files Location
- Tasks: `data/label_studio/demo_pages_tasks.json`
- Config: `data/label_studio/config.xml`
- Annotations export: `data/label_studio/demo_annotations.json`

## Stopping Label Studio
Press `Ctrl+C` in the terminal where it's running, or:
```bash
pkill -f "label-studio"
```

## Next Steps After Labeling
Once you've labeled the 7 pages:
1. Export the annotations
2. We'll analyze the patterns you used to identify RECIPE_TITLE
3. Update the heuristic relabeling script
4. Re-run on the full dataset
5. Retrain the model with better labels
