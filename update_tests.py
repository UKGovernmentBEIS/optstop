#!/usr/bin/env python3
"""Script to update test_early_stopping.py for new signatures."""

import re

def update_test_file():
    with open('tests/test_early_stopping.py', 'r') as f:
        content = f.read()

    # Pattern 1: start_task(mock_task) without epochs specified
    # Replace with start_task(mock_task, mock_samples, 5)
    content = re.sub(
        r'await\s+(\w+)\.start_task\(mock_task\)(?!\w)',
        r'await \1.start_task(mock_task, mock_samples, epochs=5)',
        content
    )

    # Pattern 2: start_task(mock_task_valid)
    content = re.sub(
        r'await\s+(\w+)\.start_task\(mock_task_valid\)(?!\w)',
        r'await \1.start_task(mock_task_valid, mock_samples_valid, epochs=5)',
        content
    )

    # Pattern 3: start_task(mock_task1) or mock_task2
    content = re.sub(
        r'await\s+(\w+)\.start_task\(mock_task(\d+)\)(?!\w)',
        r'await \1.start_task(mock_task\2, mock_samples\2, epochs=5)',
        content
    )

    with open('tests/test_early_stopping.py', 'w') as f:
        f.write(content)

    print("✅ Updated test file")
    print("Note: Tests may need additional mock_samples fixtures added")

if __name__ == '__main__':
    update_test_file()
