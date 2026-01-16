#!/bin/bash
set -e

# Configuration
LABEL_STUDIO_DIR="data/label_studio"
PORT=8080

echo "=========================================="
echo "Launching Label Studio for Demo Pages"
echo "=========================================="
echo ""
echo "Port: $PORT"
echo "Tasks: 7 demo pages (pages 69, 76, 78, 88, 90, 92, 94)"
echo ""
echo "FOCUS: Label RECIPE_TITLE (e.g., 'How to make Tea.')"
echo ""

# Set environment variables to allow local file access
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/

# Create label studio directory if it doesn't exist
mkdir -p "$LABEL_STUDIO_DIR"

# Start Label Studio with data directory
echo "Starting Label Studio on http://localhost:$PORT"
echo "=========================================="
echo ""
echo "After Label Studio starts:"
echo "  1. Open http://localhost:$PORT in your browser"
echo "  2. Create an account (first-time only)"
echo "  3. Create a new project named 'CookbookAI Demo Pages'"
echo "  4. In project settings, upload the labeling config from:"
echo "     $LABEL_STUDIO_DIR/config.xml"
echo "  5. Import tasks from:"
echo "     $LABEL_STUDIO_DIR/demo_pages_tasks.json"
echo "  6. Start labeling!"
echo ""
echo "To stop: Press Ctrl+C"
echo ""

cd "$LABEL_STUDIO_DIR"
../../.venv/bin/label-studio start \
    --port "$PORT" \
    --host 0.0.0.0
