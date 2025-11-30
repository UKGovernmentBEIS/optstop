#!/bin/bash
# Monitor test progress every 15 minutes

while true; do
    echo "=== Test Progress Check at $(date) ==="

    # Check if pytest is still running
    if ! pgrep -f "pytest.*comprehensive" > /dev/null; then
        echo "Tests completed or stopped!"
        tail -100 test_run.log
        break
    fi

    # Show recent test output
    tail -10 test_run.log | grep -E "(PASSED|FAILED|test_)"

    # Show process info
    ps aux | grep "pytest.*comprehensive" | grep -v grep | head -2

    echo "Waiting 15 minutes for next check..."
    sleep 900
done
