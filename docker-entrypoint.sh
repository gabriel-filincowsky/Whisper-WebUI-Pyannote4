#!/bin/bash
set -e

echo "Entrypoint script starting..."
echo "Checking venv status..."

# Check if venv is functional by actually testing if gradio can be imported
# When source code is mounted, a local venv directory may overlay the image's venv
# but be incomplete (missing packages or broken)
NEEDS_RESTORE=true

if [ -f "/Whisper-WebUI/venv/bin/python" ]; then
    echo "Python binary found, testing if venv is functional..."
    # Try to import gradio to verify venv is actually functional
    if /Whisper-WebUI/venv/bin/python -c "import gradio" 2>/dev/null; then
        echo "Venv is functional (gradio can be imported)"
        NEEDS_RESTORE=false
    else
        echo "Venv exists but is not functional (gradio import failed)"
    fi
else
    echo "Python binary NOT found"
fi

if [ "$NEEDS_RESTORE" = true ]; then
    echo "Venv not found or incomplete, restoring from image backup..."
    # Remove any existing incomplete venv directory
    rm -rf /Whisper-WebUI/venv
    # Copy the complete venv from backup
    echo "Copying venv from backup (this may take a moment)..."
    cp -r /venv-backup /Whisper-WebUI/venv
    echo "Venv restoration complete."
    
    # Verify restoration succeeded by testing import
    if [ ! -f "/Whisper-WebUI/venv/bin/python" ]; then
        echo "ERROR: Venv restoration failed. Python binary not found."
        exit 1
    fi
    if ! /Whisper-WebUI/venv/bin/python -c "import gradio" 2>/dev/null; then
        echo "ERROR: Venv restoration failed. Gradio cannot be imported."
        exit 1
    fi
    echo "Venv restoration verified successfully."
fi

echo "Executing application..."
# Execute the original command
exec "$@"
