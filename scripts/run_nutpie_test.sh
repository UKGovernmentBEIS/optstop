#!/bin/bash
# Helper script to run the enhanced large-scale bridge test with nutpie tracking

echo "======================================================================"
echo "Large-Scale Bridge Test with Nutpie Tracking"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Verify nutpie installation"
echo "  2. Backup previous test results"
echo "  3. Run all 3 datasets with nutpie tracking"
echo "  4. Generate performance comparison report"
echo ""

# Navigate to project root
cd /home/ubuntu/optstop

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    source venv/bin/activate
fi

# Check nutpie availability
echo "🔍 Checking nutpie availability..."
python -c "
from optstop import gpu_utils
available, version = gpu_utils.check_nutpie_available()
if available:
    print(f'✅ Nutpie IS available (version {version})')
    print(f'   Expected: 2-5× CPU speedup for all inference pathways')
else:
    print('⚠️  Nutpie NOT available')
    print('   Tests will use PyMC default sampler (slower)')
    print('   To install: pip install nutpie')
" || exit 1

echo ""
read -p "Press Enter to start the test (or Ctrl+C to cancel)..."
echo ""

# Run the enhanced test
echo "🚀 Starting large-scale bridge test..."
echo ""
PYTHONPATH=/home/ubuntu/optstop python scripts/test_large_datasets_bridge_with_nutpie.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ TEST COMPLETED SUCCESSFULLY"
    echo "======================================================================"
    echo ""
    echo "Results saved to: test_outputs/large_scale/"
    echo ""
    echo "To view the comparison report:"
    echo "  python scripts/generate_comparison_report.py"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "❌ TEST FAILED"
    echo "======================================================================"
    echo ""
    echo "Check logs in test_outputs/large_scale/ for details"
    echo ""
    exit 1
fi
