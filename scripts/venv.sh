#!/bin/bash

# Configuration
VENV_DIR=".venv"

# 1. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    echo "Virtual environment created."
fi

# 2. Check if the script is being sourced (required for activation)
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0

if [ $SOURCED -eq 1 ]; then
    # Script is sourced, activate the environment
    source "$VENV_DIR/bin/activate"
    echo "✅ Environment activated! You are now in ($VENV_DIR)."
else
    # Script is executed directly, warn the user but still create venv if needed
    echo "⚠️  WARNING: You ran this script as an executable."
    echo "The environment will NOT be activated in your current shell."
    echo ""
    echo "👉 To activate, run command:"
    echo "   source scripts/venv.sh"
    echo ""
fi
