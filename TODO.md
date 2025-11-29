# optstop Development TODO

## Immediate Tasks

1. **Confirm latest large scale test script has run correctly**
   - Run `scripts/test_large_datasets_numpyro.py` with v0.2.1 parameters
   - Verify random_seed=42 is logged and included in diagnostics
   - Verify entropy_stabilization_threshold=0.001 produces more conservative ordinal stopping
   - Check all three inference pathways route correctly

2. **Cleanup unnecessary files from development into Q4_Development_Files/**
   - Move outdated test scripts and development artifacts
   - Organize documentation drafts and planning files
   - Keep root directory clean with only production-ready files

3. **Start working through logging and print statement refinement**
   - Review logging levels (DEBUG, INFO, WARNING)
   - Ensure consistent log message formatting
   - Remove or reduce verbose print statements in production code
   - Add structured logging where appropriate
