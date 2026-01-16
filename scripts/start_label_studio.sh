#!/bin/bash

# Configuration
PORT=8080

echo "=========================================="
echo "Label Studio for CookbookAI Demo Pages"
echo "=========================================="
echo ""
echo "Starting on http://localhost:$PORT"
echo ""

# Set environment variables to allow local file access
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/

# Start Label Studio
.venv/bin/label-studio start --port "$PORT"
