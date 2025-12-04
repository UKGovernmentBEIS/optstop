# optstop Development TODO

## Immediate Tasks

1. ~~**Run large-scale test with v0.2.2 sampler fixes**~~ ✅ COMPLETE (2024-12-01)
   - All three inference pathways verified working with PyMC default on CPU
   - random_seed=42 and entropy_stabilization_threshold=0.001 correctly applied
   - Results in `test_outputs/large_scale/` with full diagnostics

2. ~~**Cleanup unnecessary files from development into Q4_Development_Files/**~~ ✅ COMPLETE
   - Moved 24 files (dev docs, test scripts, logs) to Q4_Development_Files/
   - Main folder now contains only essential package files and documentation

3. ~~**Start working through logging and print statement refinement**~~ ✅ COMPLETE (2024-12-01)
   - Reviewed and updated logging in bridge-related files:
     - `early_stopping.py`: Added trace_message() for inspect_ai integration, commented out verbose logs
     - `rule.py`: Commented out live mode stopping logs (results in output), TIMING_TEST logs
     - `ordinal_model.py`: Removed investigation logging, kept error handling
     - `ordinal_utils.py`: Added trace_message() for key warnings, streamlined type determination logs
   - `gpu_utils.py`: Kept all logging (GPU detection/configuration is valuable)
   - Reference: `LOGGING_AUDIT.md` for original analysis

4. **Polish documentation (README.md, BRIDGE_API_REFERENCE.md)**
   - Review for consistency and clarity
   - Ensure all code examples are accurate and tested
   - Verify version numbers and compatibility statements are up to date
   - Check for any outdated information or stale references
   - Consider adding more practical examples for common use cases

## v0.2.2 Large-Scale Test Results (2024-12-01)

| Dataset | Type | Trials Ran | Efficiency | Runtime | Group Stops |
|---------|------|------------|------------|---------|-------------|
| 1 | Binary Discrete | 3625/5000 | 27.5% | 8.85 min | 2/5 |
| 2 | Ordinal Discrete | 2875/5000 | 42.5% | 865 min (14.4 hrs) | 3/5 |
| 3 | Continuous Bounded | 2325/5000 | 53.5% | 8.76 min | 4/5 |

**Verification Summary:**
- ✓ Binary pathway: CI width stopping, stabilization checks
- ✓ Ordinal pathway: Hybrid inference (modal_ci_narrow_validated + entropy_stabilized)
- ✓ Continuous pathway: Hierarchical CI width + stabilization
- ✓ Random seed propagation and logging
- ✓ Group-level stopping with detailed diagnostics

**Ordinal Performance Note:**
Ordinal inference (14.4 hrs) is significantly slower than binary/continuous (~9 min).
This is due to the computational cost of entropy-based Bayesian inference required
for valid stopping decisions on non-peaked distributions. Alternative backends
(numpyro/JAX) were tested but showed no improvement on CPU. The hybrid approach
(fast modal pathway for peaked distributions, entropy pathway otherwise) represents
the best balance of validity and performance.

## Recent Changes (v0.2.2)

- **Sampler selection simplified**: All score types now use PyMC default on CPU
- **Benchmarks showed**: numpyro/JAX is ~26% slower than PyMC on CPU
- **GPU path unchanged**: Still uses numpyro for acceleration when available
- **Test datasets regenerated**: Now use 1-indexed epochs (matching inspect_ai)
- **value_to_float import**: Now cached at initialization (no repeated warnings)
