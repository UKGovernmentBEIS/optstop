#!/usr/bin/env python3
"""
Quick validation script to verify ordinal diagnostic transparency improvements.

This script checks existing test output to see what fields are present,
and provides guidance on what will be added after the next test run.
"""

import json
import sys
from pathlib import Path

def validate_diagnostics():
    """Validate ordinal diagnostic improvements in test output."""

    print("="*80)
    print("ORDINAL DIAGNOSTIC TRANSPARENCY VALIDATION")
    print("="*80)
    print()

    # Check if test output exists
    output_file = Path("test_outputs/large_scale/dataset_2_diagnostics.json")
    if not output_file.exists():
        print(f"❌ Test output not found: {output_file}")
        print("   Run the test first to generate diagnostics.")
        return False

    # Load diagnostics
    with open(output_file) as f:
        data = json.load(f)

    print("📁 Loaded:", output_file)
    print()

    # Check stabilization histories
    print("-"*80)
    print("STABILIZATION HISTORIES CHECK")
    print("-"*80)

    histories = data.get('diagnostics', {}).get('stabilization_histories', {})
    if not histories:
        print("❌ No stabilization histories found")
        return False

    print(f"Found {len(histories)} groupings")
    print()

    # Check for new ordinal fields
    ordinal_fields_found = {
        'final_modal_ci_width': 0,
        'final_entropy': 0,
        'final_entropy_threshold': 0,
        'final_modal_ci': 0,
        'final_entropy_ci_width': 0,
        'final_relative_change': 0,
        'ordinal_pathway': 0
    }

    for grouping, hist in histories.items():
        for field in ordinal_fields_found.keys():
            if field in hist and hist[field] is not None:
                ordinal_fields_found[field] += 1

    print("Ordinal-specific fields in stabilization_histories:")
    all_present = True
    for field, count in ordinal_fields_found.items():
        if count > 0:
            print(f"  ✅ {field}: Present in {count}/{len(histories)} groupings")
        else:
            print(f"  ⚠️  {field}: Not present (will be added after rerun)")
            all_present = False
    print()

    # Check for glossary
    print("-"*80)
    print("ORDINAL GLOSSARY CHECK")
    print("-"*80)

    glossary = data.get('diagnostics', {}).get('ordinal_glossary')
    if glossary:
        print(f"✅ Ordinal glossary present with {len(glossary)} entries:")
        for reason, info in glossary.items():
            print(f"  - {reason}")
            print(f"    Description: {info.get('description', 'N/A')[:70]}...")
    else:
        print("⚠️  Ordinal glossary not present (will be added after rerun)")
        all_present = False
    print()

    # Check group stopping events
    print("-"*80)
    print("GROUP-LEVEL STOPPING EVENTS")
    print("-"*80)

    group_stopping = data.get('group_level_stopping', {})
    stopped_groupings = group_stopping.get('stopped_groupings', [])

    if stopped_groupings:
        print(f"Found {len(stopped_groupings)} stopping events:")
        for event in stopped_groupings[:3]:  # Show first 3
            print(f"  {event}")
        if len(stopped_groupings) > 3:
            print(f"  ... and {len(stopped_groupings) - 3} more")
        print()

        # Check if events have metrics
        has_metrics = any('(modal CI width=' in event or 'entropy=' in event for event in stopped_groupings)
        if has_metrics:
            print("✅ Events contain ordinal metrics")
        else:
            print("⚠️  Events show '(no metrics)' - will be fixed after rerun")
    else:
        print("No group stopping events found")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if all_present:
        print("✅ ALL IMPROVEMENTS DETECTED!")
        print("   The ordinal diagnostic transparency fixes are working correctly.")
    else:
        print("⚠️  IMPROVEMENTS NOT YET VISIBLE")
        print("   This is expected if testing existing diagnostics from before the code changes.")
        print()
        print("To see the improvements:")
        print("  1. Run: PYTHONPATH=/home/ubuntu/optstop python scripts/test_logging_fix.py")
        print("  2. Check the log output for improved metrics")
        print("  3. Re-run this validation script")

    print()
    return all_present

if __name__ == "__main__":
    success = validate_diagnostics()
    sys.exit(0 if success else 1)
