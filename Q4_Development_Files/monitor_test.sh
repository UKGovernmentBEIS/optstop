#!/bin/bash
# Monitor test battery progress

echo "=== Test Battery Progress Monitor ==="
echo "Current time: $(date)"
echo ""

# Count completed scenarios
completed=$(grep -c "^RESULTS:" test_run_output.log)
echo "Completed scenarios: $completed/20"
echo ""

# Show last scenario
echo "=== Current/Last Scenario ==="
grep -E "(SCENARIO [0-9]+/20|TEST SCENARIO:)" test_run_output.log | tail -2
echo ""

# Show recent results
echo "=== Last 3 Completed Results ==="
grep -A 6 "^RESULTS:" test_run_output.log | tail -21
echo ""

# Show any errors
errors=$(grep -c "ERROR" test_run_output.log)
if [ $errors -gt 0 ]; then
    echo "=== ERRORS FOUND: $errors ==="
    grep "ERROR" test_run_output.log | tail -5
    echo ""
fi

# Estimate completion
if [ $completed -gt 0 ]; then
    # Get timestamp of first result
    lines=$(wc -l < test_run_output.log)
    echo "Log file lines: $lines"
    echo ""
fi

echo "=== To watch live output ==="
echo "tail -f test_run_output.log"
