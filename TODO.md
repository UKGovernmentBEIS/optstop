# optstop Development TODO

## Immediate Tasks

1. **Run large-scale test with v0.2.2 sampler fixes**
   - Run `scripts/test_large_datasets_numpyro.py` (now uses PyMC default for ALL score types on CPU)
   - Expected performance improvement: ordinal inference ~20-40% faster than numpyro
   - Verify all three inference pathways work correctly with PyMC default
   - Check random_seed=42 and entropy_stabilization_threshold=0.001 are applied

2. ~~**Cleanup unnecessary files from development into Q4_Development_Files/**~~ ✅ COMPLETE
   - Moved 24 files (dev docs, test scripts, logs) to Q4_Development_Files/
   - Main folder now contains only essential package files and documentation

3. **Start working through logging and print statement refinement**
   - Review logging levels (DEBUG, INFO, WARNING)
   - Ensure consistent log message formatting
   - Remove or reduce verbose print statements in production code
   - Add structured logging where appropriate
   - Reference: `LOGGING_AUDIT.md` for previous analysis

## Recent Changes (v0.2.2)

- **Sampler selection simplified**: All score types now use PyMC default on CPU
- **Benchmarks showed**: numpyro/JAX is ~26% slower than PyMC on CPU
- **GPU path unchanged**: Still uses numpyro for acceleration when available
- **Test datasets regenerated**: Now use 1-indexed epochs (matching inspect_ai)
- **value_to_float import**: Now cached at initialization (no repeated warnings)
