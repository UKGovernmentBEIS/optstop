#!/bin/bash
# Setup script to install inspect_ai from the feature/early-stopping branch
# This creates an isolated virtual environment for the integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTSTOP_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_DIR="$SCRIPT_DIR/venv_inspect"

echo "=============================================="
echo "Setting up inspect_ai feature/early-stopping branch"
echo "=============================================="
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Optstop directory: $OPTSTOP_DIR"
echo "Virtual env: $VENV_DIR"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "Installing inspect_ai from feature/early-stopping branch..."
pip install --upgrade pip -q

# Install inspect_ai from the feature/early-stopping branch
pip install "git+https://github.com/UKGovernmentBEIS/inspect_ai.git@feature/early-stopping" -q

# Install optstop dependencies (but not optstop itself - we'll use PYTHONPATH)
pip install numpy pandas pymc arviz scipy pydantic tqdm -q

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To use this environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the synthetic log generator:"
echo "  PYTHONPATH=$OPTSTOP_DIR:\$PYTHONPATH python $SCRIPT_DIR/generate_authentic_eval_logs.py"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "
from inspect_ai.util import EarlyStopping, EarlyStop, EarlyStoppingSummary
from inspect_ai.log import EvalLog, EvalResults, write_eval_log
print('  - EarlyStopping protocol: OK')
print('  - EarlyStop model: OK')
print('  - EarlyStoppingSummary model: OK')
print('  - EvalLog model: OK')
print('  - write_eval_log function: OK')
print('')
print('All imports successful!')
"
