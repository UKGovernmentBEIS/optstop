#!/bin/bash
# Monitor corrected test suite every 15 minutes

LOGFILE="test_run_corrected.log"
CHECK_INTERVAL=900  # 15 minutes

while true; do
    echo "=== Check at $(date) ==="

    # Check if pytest process is still running
    if ! pgrep -f "pytest.*test_early_stopping_comprehensive" > /dev/null; then
        echo "Tests completed!"
        echo ""
        echo "Final results:"
        tail -100 "$LOGFILE"
        break
    fi

    # Show recent progress
    echo "Tests still running. Recent output:"
    tail -15 "$LOGFILE" | grep -E "(PASSED|FAILED|test_)" || echo "  (No recent test completions)"
    echo ""
    echo "Next check in 15 minutes..."
    sleep $CHECK_INTERVAL
done
