from __future__ import annotations

from typing import Any, Optional
try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.10-3.11
import pandas as pd
import numpy as np
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from pydantic import BaseModel, Field, JsonValue

# Import GPU utilities for configuration
from . import gpu_utils

# Import inspect_ai classes - with fallback to mock for testing
try:
    from inspect_ai.dataset._dataset import Sample
    from inspect_ai.log._log import EvalSpec
    from inspect_ai.scorer._metric import SampleScore
    from inspect_ai.util import EarlyStopping, trace_message
    from inspect_ai.util._early_stopping import EarlyStop
except (ImportError, AttributeError):
    # trace_message fallback for testing
    def trace_message(logger, component, message):
        logger.info(f"[{component}] {message}")
    # inspect_ai is not installed - fall back to the in-package protocol stubs so the
    # core package still imports (the inspect_ai bridge itself requires inspect_ai).
    from ._mock_inspect_early_stop import EarlyStopping, EarlyStop

    # Create minimal mocks for other types
    class Sample:
        def __init__(self, id, metadata=None):
            self.id = id
            self.metadata = metadata or {}

    class EvalSpec:
        def __init__(self, model, task, eval_id=None, metadata=None, tags=None):
            self.model = model
            self.task = task
            self.eval_id = eval_id
            self.metadata = metadata or {}
            self.tags = tags or []

    class MockScore:
        def __init__(self, value):
            self.value = value

    class SampleScore:
        def __init__(self, score):
            if hasattr(score, 'value'):
                self.score = score
            else:
                self.score = MockScore(score)

# Configure logger
logger = logging.getLogger(__name__)


class StoppedSample(BaseModel):
    """Record of early stop for a sample/epoch.

    This is an internal tracking class used by OptimalStoppingManager
    to maintain records of stopped samples for diagnostic purposes.
    """

    id: str | int
    """Sample dataset id."""

    epoch: int
    """Sample epoch."""

    early_stop: EarlyStop
    """Early stop directive."""


class OptimalStoppingManager(EarlyStopping):
    """Optimal stopping manager for inspect_ai evaluations using optstop.

    This class implements the EarlyStopping protocol and provides Bayesian optimal
    stopping for sample evaluation across epochs.

    Workflow:
        1. start_task(): Initialize compiled_dataset with all planned trials
        2. schedule_sample(): Fast lookup to check if sample should run (called before each trial)
        3. complete_sample(): Update scores, run periodic stopping checks
           - Calls _run_stopping_inference() every reanalysis_interval samples
           - Each call runs BOTH sample-level AND group-level checks automatically
           - Updates schedule_status for samples/groups that meet stopping criteria
        4. complete_task(): Generate final diagnostics
           - No additional inference needed (already ran during complete_sample)
           - Returns comprehensive diagnostics and efficiency metrics

    Stabilization History:
        - Maintained per-grouping in _stabilization_histories dict
        - Tracks CI widths, slopes, and entropy across inference calls
        - Enables stabilization criteria to work correctly across multiple calls
        - Passed into and returned from optimal_stopping_live_single()
        - Group-level checks append to history each time they run
    """

    # Protocol-required attribute from EarlyStopping
    compiled_dataset: Optional[pd.DataFrame]

    # Internal column names for compiled_dataset (not user-configurable)
    _SCORE_COLUMN = "score"
    _SAMPLE_ID_COLUMN = "sample_id"
    _EPOCH_COLUMN = "epoch"

    def __init__(
        self,
        optstop_params: dict[str, Any],
        grouping_columns: list[str],
        reanalysis_interval: int = 10,
        min_samples_per_grouping: int = 5,
        ordinal_tasks: Optional[list[str]] = None,
        ordinal_max_score: int = 10,
        ordinal_inference: str = 'hybrid',
        ordinal_model_type: str = 'ordered_logistic',
        gpu_ids: Optional[list[int]] = None,
        entropy_threshold: float = 0.8,
        prior_mu: float = 0.0,
        prior_sigma: Optional[float] = None,
        manager_name: str = "optstop",
        shadow_mode: bool = False,
        score_choice: Optional[str] = None,
        score_value_key: Optional[str] = None,
        score_agg: Optional[str] = None,
        random_seed: Optional[int] = None,
        use_preallocation: bool = True,  # Fix 1: Pre-allocation flag
    ):
        """Initialize optimal stopping manager.

        Args:
            optstop_params: Dictionary of optimal stopping parameters.
                Core stopping parameters:
                    - delta_item: Effect size threshold for item-level stopping
                    - delta_cap: Effect size threshold for capability-level stopping
                    - cred_level: Credible interval level (e.g., 0.97)
                    - conservatism: Conservatism factor for stopping decisions
                MCMC sampling parameters (optional, with sensible defaults):
                    - draws: Number of posterior samples (default: 1000 CPU, 2000 GPU)
                    - tune: Number of tuning samples (default: 1000 CPU, 2000 GPU)
                    - chains: Number of MCMC chains (default: 4)
                    - cores: Number of CPU cores for sampling (default: 4)
                    - target_accept: Target acceptance rate for NUTS sampler
                      (default: 0.90 CPU, 0.95 GPU). Higher values (e.g., 0.95-0.99)
                      reduce divergences but increase computation time. Lower values
                      (e.g., 0.80-0.90) are faster but may have more divergences.
            grouping_columns: List of columns to use for grouping decisions.
                REQUIRED - user must specify.
                Can reference any column in compiled_dataset (from EvalSpec or sample metadata).
                Examples: ['model', 'task'], ['model', 'difficulty'], ['dataset', 'temperature']
            reanalysis_interval: Run inference every N completed samples
            min_samples_per_grouping: Minimum completed samples before first inference
            ordinal_tasks: Task names/substrings using ordinal scoring
            ordinal_max_score: Maximum ordinal score value
            ordinal_inference: Ordinal inference mode ('modal', 'entropy', 'hybrid')
            ordinal_model_type: Hierarchical model type for ordinal inference
                ('ordered_logistic' or 'dirichlet'). Default: 'ordered_logistic'
            gpu_ids: GPU IDs for computation (non-empty list enables GPU detection)
            entropy_threshold: Proportion of max entropy for false peak detection in
                ordinal hybrid mode (0 to 1). Scaled internally by log2(num_categories).
                Lower values require more peaked distributions; higher values are more
                permissive. Default: 0.8.
            prior_mu: Centre of the Normal prior on mu_group (logit scale). Default 0.0
                corresponds to 50% on the probability scale (assumption-free default).
                Positive values bias toward higher performance, negative toward lower.
                Useful for users with domain-specific performance expectations.
            prior_sigma: Scale (standard deviation) of the Normal prior on mu_group.
                If None (default), uses pathway-specific defaults: 1.5 for binary/continuous,
                2.0 for ordinal. If explicitly set, applies to all pathways.
                Controls how diffuse the prior is on the logit scale. Smaller values
                provide stronger regularization toward prior_mu.
            manager_name: Name identifier for this manager
            shadow_mode: If True, schedule_sample() always returns None (run all trials).
                Inference still runs and stopping decisions are recorded. complete_task()
                diagnostics include stopped_at_trial_count and shadow_mode_summary.
            score_choice: Scorer name to select from the scores dict (outer dict key).
                If None, uses first score in dict. Mutually exclusive with score_agg.
            score_value_key: Key to extract from dict-valued Score.value objects.
                Many inspect_ai scorers return Score.value as a dict (e.g.,
                {"healthbench_score": 0.35, "criteria_met": 5}). This parameter
                specifies which key holds the numeric value for inference.
                If Score.value is already a scalar, this parameter is ignored.
                If Score.value is a dict and this parameter is None, the sample
                is skipped with a warning.
            score_agg: Aggregation method for multiple scores ('mean', 'median', 'mode', 'max').
                If None, uses single score. Mutually exclusive with score_choice.
            random_seed: Random seed for MCMC sampling reproducibility.
                If None, a seed is auto-generated and logged for reproducibility tracking.
            use_preallocation: Enable model pre-allocation at max_n_items (Fix 1).
                When True (default), models are pre-allocated to avoid recompilation
                as n_items grows. When False, uses original dynamic behavior.
                Set to False to revert if issues arise.

        Raises:
            ValueError: If both score_choice and score_agg are specified (mutually exclusive).
        """
        # Validate score extraction parameters
        if score_choice is not None and score_agg is not None:
            raise ValueError(
                "score_choice and score_agg are mutually exclusive. "
                "Specify only one or leave both as None."
            )

        if score_agg is not None and score_agg not in ['mean', 'median', 'mode', 'max']:
            raise ValueError(
                f"score_agg must be one of ['mean', 'median', 'mode', 'max'], got '{score_agg}'"
            )

        # Configuration
        self.optstop_params = optstop_params.copy()  # Copy to avoid modifying original
        self.grouping_columns = grouping_columns
        self.reanalysis_interval = reanalysis_interval
        self.min_samples_per_grouping = min_samples_per_grouping
        self.manager_name = manager_name
        self.entropy_threshold = entropy_threshold
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.shadow_mode = shadow_mode
        self.score_choice = score_choice
        self.score_value_key = score_value_key
        self.score_agg = score_agg

        # Random seed handling: generate if not provided, always store for reproducibility
        if random_seed is not None:
            self.random_seed = int(random_seed)  # Ensure Python int for JSON serialization
            self._seed_source = "user_specified"
        else:
            # Generate a random seed using system entropy
            self.random_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
            self._seed_source = "auto_generated"

        # Add seed to optstop_params so it propagates to MCMC sampling
        self.optstop_params['random_seed'] = self.random_seed

        # Log seed immediately for reproducibility tracking
        trace_message(logger, "OptimalStopping", f"Random seed: {self.random_seed} ({self._seed_source})")

        # Ordinal configuration
        self.ordinal_tasks = ordinal_tasks
        self.ordinal_max_score = ordinal_max_score
        self.ordinal_inference = ordinal_inference
        self.ordinal_model_type = ordinal_model_type

        # GPU configuration
        self.gpu_ids = gpu_ids

        # Pre-allocation flag (Fix 1 for scaling issue)
        # When True, models are pre-allocated at max_n_items to avoid recompilation
        # Set to False to revert to original behavior if issues arise
        self._use_preallocation = use_preallocation

        # Data tracking
        self.compiled_dataset: Optional[pd.DataFrame] = None
        self.stopped_samples: list[StoppedSample] = []

        # Per-grouping decision counters for consistent inference timing
        self._decision_counters: dict[str, int] = {}

        # Track stopped samples per grouping to prevent duplicate logging
        self._stopped_sample_ids: dict[str, set] = {}

        # Track stopped groupings to prevent duplicate logging
        self._stopped_groupings: set[str] = set()

        # Track when stopping was first triggered per grouping
        self._stopped_at_trial_count: dict[str, dict] = {}

        # Cache for fast lookups
        self._schedule_cache: dict[tuple, bool] = {}

        # Stabilization histories per grouping
        self._stabilization_histories: dict[str, dict[str, list[float]]] = {}

        # PyMC model caches per grouping (OPTIMIZATION #2)
        # Persists PyMC model compilation across inference calls to avoid recompilation
        # Separate caches for item-level and group-level models
        self._binary_item_model_caches: dict[str, dict[str, Any]] = {}
        self._binary_group_model_caches: dict[str, dict[str, Any]] = {}
        self._ordinal_item_model_caches: dict[str, dict[str, Any]] = {}
        self._ordinal_group_model_caches: dict[str, dict[str, Any]] = {}
        self._continuous_item_model_caches: dict[str, dict[str, Any]] = {}
        self._continuous_group_model_caches: dict[str, dict[str, Any]] = {}

        # Initialize dedicated executor for inference operations
        # Uses max_workers=1 to ensure sequential inference per manager
        # (PyMC creates its own multiprocessing pool internally with 4 chains)
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"optstop_inference_{manager_name}"
        )

        # Track pending inference per grouping (Fix 7: Skip redundant queued inference)
        # When a newer inference request arrives while one is pending, the older one
        # would produce stale results (the newer request has strictly more data).
        # This dict tracks which groupings have pending inference to skip redundant calls.
        self._pending_inference: dict[str, Any] = {}

        # Initialize score value converter (cached from inspect_ai import)
        # This avoids repeated import attempts in _extract_score_value()
        self._value_converter = self._init_value_converter()

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # 1. Validate parameter ranges (all are optional with defaults)
        # Note: optimal_stopping_live_single() provides defaults for these

        # Stopping thresholds
        delta_item = self.optstop_params.get('delta_item')  # default: 0.05
        if delta_item is not None and delta_item <= 0:
            raise ValueError(f"delta_item must be > 0, got {delta_item}")

        delta_cap = self.optstop_params.get('delta_cap')  # default: 0.05
        if delta_cap is not None and delta_cap <= 0:
            raise ValueError(f"delta_cap must be > 0, got {delta_cap}")

        # Statistical parameters
        cred_level = self.optstop_params.get('cred_level')  # default: 0.97
        if cred_level is not None and not (0 < cred_level < 1):
            raise ValueError(f"cred_level must be between 0 and 1, got {cred_level}")

        conservatism = self.optstop_params.get('conservatism')  # default: 5
        if conservatism is not None and conservatism < 1:
            raise ValueError(f"conservatism must be >= 1, got {conservatism}")

        low_perf_threshold = self.optstop_params.get('low_performance_threshold')  # default: 0.01
        if low_perf_threshold is not None and not (0 <= low_perf_threshold <= 1):
            raise ValueError(f"low_performance_threshold must be between 0 and 1, got {low_perf_threshold}")

        # Stabilization parameters
        CI_delta = self.optstop_params.get('CI_delta')  # default: 0.00001
        if CI_delta is not None and CI_delta <= 0:
            raise ValueError(f"CI_delta must be > 0, got {CI_delta}")

        stab_window = self.optstop_params.get('stab_window')  # default: 15
        if stab_window is not None and stab_window <= 0:
            raise ValueError(f"stab_window must be > 0, got {stab_window}")

        rep_batch_size = self.optstop_params.get('rep_batch_size')  # default: 1
        if rep_batch_size is not None and rep_batch_size <= 0:
            raise ValueError(f"rep_batch_size must be > 0, got {rep_batch_size}")

        # Entropy convergence threshold for ordinal hybrid stopping (Pathway 2)
        # Accepts both new name and deprecated old name for backward compatibility
        entropy_conv = self.optstop_params.get('entropy_convergence_threshold')
        entropy_stab_deprecated = self.optstop_params.get('entropy_stabilization_threshold')
        if entropy_conv is not None and entropy_stab_deprecated is not None:
            logger.warning(
                "Both entropy_convergence_threshold and entropy_stabilization_threshold provided; "
                "using entropy_convergence_threshold. Remove entropy_stabilization_threshold from your config."
            )
            del self.optstop_params['entropy_stabilization_threshold']
        elif entropy_conv is None and entropy_stab_deprecated is not None:
            logger.warning(
                "entropy_stabilization_threshold is deprecated; use entropy_convergence_threshold instead. "
                "Note: the mechanism has changed from relative-change (default 0.002) to absolute entropy "
                "CI width on [0,1] scale (default 0.10)."
            )
            if entropy_stab_deprecated < 0.01:
                logger.warning(
                    f"Your value {entropy_stab_deprecated} appears to be on the old relative-change scale. "
                    f"The new default is 0.10 (absolute entropy CI width on [0,1] scale). "
                    f"Consider removing this parameter to use the new default."
                )
            entropy_conv = entropy_stab_deprecated
            del self.optstop_params['entropy_stabilization_threshold']
        if entropy_conv is not None and entropy_conv <= 0:
            raise ValueError(f"entropy_convergence_threshold must be > 0, got {entropy_conv}")
        if entropy_conv is not None:
            self.optstop_params['entropy_convergence_threshold'] = entropy_conv

        # PyMC sampling parameters (passed to sampling_kwargs)
        tune = self.optstop_params.get('tune')  # default: auto-configured by gpu_utils
        if tune is not None and tune < 0:
            raise ValueError(f"tune must be >= 0, got {tune}")

        draws = self.optstop_params.get('draws')  # default: auto-configured by gpu_utils
        if draws is not None and draws <= 0:
            raise ValueError(f"draws must be > 0, got {draws}")

        # 3. Validate grouping_columns
        if not self.grouping_columns:
            raise ValueError(
                "grouping_columns cannot be empty. Must specify at least one column "
                "(e.g., ['model'] or ['model', 'task'])"
            )

        # Validate grouping column format (Fix #3 integrated here)
        for col in self.grouping_columns:
            if col not in ['model', 'task']:
                # Check if it matches metadata.* or tag.* pattern
                if not (col.startswith('metadata.') or col.startswith('tag.')):
                    raise ValueError(
                        f"Invalid grouping column '{col}'. "
                        f"Must be 'model', 'task', 'metadata.<key>', or 'tag.<name>'"
                    )
                # Validate there's content after the prefix
                if col.startswith('metadata.') and len(col) <= len('metadata.'):
                    raise ValueError(f"Invalid grouping column '{col}': missing key after 'metadata.'")
                if col.startswith('tag.') and len(col) <= len('tag.'):
                    raise ValueError(f"Invalid grouping column '{col}': missing tag name after 'tag.'")

        # 4. Validate intervals
        if self.reanalysis_interval <= 0:
            raise ValueError(f"reanalysis_interval must be > 0, got {self.reanalysis_interval}")

        if self.min_samples_per_grouping < 0:
            raise ValueError(
                f"min_samples_per_grouping must be >= 0, got {self.min_samples_per_grouping}"
            )

        # 5. Validate ordinal configuration
        if self.ordinal_inference not in ['modal', 'entropy', 'hybrid']:
            raise ValueError(
                f"ordinal_inference must be one of ['modal', 'entropy', 'hybrid'], "
                f"got '{self.ordinal_inference}'"
            )

        if self.ordinal_max_score <= 0:
            raise ValueError(f"ordinal_max_score must be > 0, got {self.ordinal_max_score}")


    def _init_value_converter(self) -> Any:
        """Initialize and cache the value_to_float converter from inspect_ai.

        This method attempts to import inspect_ai's value_to_float function once
        during initialization, avoiding repeated import attempts in _extract_score_value().

        Returns:
            The value_to_float converter callable if available, None otherwise.
        """
        try:
            from inspect_ai.scorer._metric import value_to_float
            converter = value_to_float()
            # logger.debug("Successfully imported value_to_float from inspect_ai")  # Superseded by version log below
            return converter
        except ImportError:
            logger.info(
                "Could not import value_to_float from inspect_ai. "
                "Using basic float conversion for score extraction."
            )
            return None
        except Exception as e:
            trace_message(
                logger, "OptimalStopping",
                f"Error initializing value_to_float converter: {e}. Using basic float conversion."
            )
            return None

    def _print_configuration_summary(self, num_samples: int, num_epochs: int) -> None:
        """Print comprehensive configuration summary to console.

        Args:
            num_samples: Number of samples in the dataset
            num_epochs: Number of epochs per sample
        """
        print("\n" + "="*80)
        print(f"OptimalStoppingManager Configuration Summary ({self.manager_name})")
        print("="*80)

        # Dataset Configuration
        print("\nDataset Configuration:")
        print(f"  • Samples: {num_samples}")
        print(f"  • Epochs per sample: {num_epochs}")
        print(f"  • Total planned trials: {num_samples * num_epochs}")
        print(f"  • Grouping columns: {', '.join(self.grouping_columns)}")

        # Stopping Parameters (from optstop_params)
        print("\nOptimal Stopping Parameters:")
        params_to_show = {
            'delta_item': ('Item CI width threshold', 0.05),
            'delta_cap': ('Grouping CI width threshold', 0.05),
            'cred_level': ('Credibility level', 0.97),
            'conservatism': ('Conservatism factor', 5),
            'low_performance_threshold': ('Low performance threshold', 0.01),
            'CI_delta': ('CI stabilization slope threshold', 0.00001),
            'stab_window': ('Stabilization window', 15),
            'rep_batch_size': ('Repetition batch size', 1),
            'entropy_convergence_threshold': ('Entropy convergence threshold (P2)', 0.10),
            'draws': ('MCMC draws', 1000),
            'tune': ('MCMC tune steps', 1000),
            'chains': ('MCMC chains', 4),
            'cores': ('CPU cores', 4)
        }

        for key, (label, default) in params_to_show.items():
            value = self.optstop_params.get(key, default)
            print(f"  • {label}: {value}")
        print(f"  • Prior mu (group-level): {self.prior_mu}")
        prior_sigma_display = self.prior_sigma if self.prior_sigma is not None else "pathway-specific (binary/cont: 1.5, ordinal: 2.0)"
        print(f"  • Prior sigma (group-level): {prior_sigma_display}")

        # Inference Control
        print("\n  Inference Control:")
        print(f"  • Reanalysis interval: every {self.reanalysis_interval} completed samples")
        print(f"  • Min samples per grouping: {self.min_samples_per_grouping}")
        if self.shadow_mode:
            print("  • Shadow mode: ENABLED (all trials will run, stopping disabled)")
        else:
            print("  • Shadow mode: Disabled (normal stopping behavior)")

        # Score Extraction Configuration
        print("\n Score Extraction:")
        if self.score_choice is not None:
            print(f"  • Mode: Extract specific score by key")
            print(f"  • Score key: '{self.score_choice}'")
        elif self.score_agg is not None:
            print(f"  • Mode: Aggregate all scores")
            print(f"  • Aggregation method: {self.score_agg}")
        else:
            print("  • Mode: Default (use first score from dict)")
        if self.score_value_key is not None:
            print(f"  • Dict value key: '{self.score_value_key}'")

        # Ordinal Configuration
        print("\n Ordinal Scoring Configuration:")
        if self.ordinal_tasks:
            print(f"  • Ordinal tasks: {', '.join(self.ordinal_tasks)}")
            print(f"  • Max ordinal score: {self.ordinal_max_score}")
            print(f"  • Inference mode: {self.ordinal_inference}")
            print(f"  • Entropy threshold: {self.entropy_threshold} (proportion of max entropy)")
        else:
            print("  • Ordinal tasks: None (binary scoring only)")

        # GPU Configuration
        print("\n Hardware Configuration:")
        if self.gpu_ids and len(self.gpu_ids) > 0:
            print(f"  • GPU IDs: {self.gpu_ids}")
            # Check actual GPU availability
            gpu_available, gpu_backend, _ = gpu_utils.check_gpu_availability()
            if gpu_available:
                print(f"  • GPU status: Available ({gpu_backend})")
            else:
                print("  • GPU status: Requested but not available (falling back to CPU)")
        else:
            print("  • GPU: Disabled (CPU-only mode)")

        # Reproducibility Configuration
        print("\n Reproducibility:")
        print(f"  • Random seed: {self.random_seed}")
        print(f"  • Seed source: {self._seed_source}")

        print("\n" + "="*80 + "\n")

    def _get_grouping_values_for_sample(self, sample_id: str | int) -> dict[str, Any]:
        """Extract grouping values for a sample from compiled_dataset.

        All grouping columns (from EvalSpec or sample metadata) are now in the
        compiled_dataset, so we can extract them directly by looking up the sample.

        Handles grouping column format translation:
        - 'model', 'task' → direct column lookup
        - 'metadata.<key>' → looks up column '<key>' (metadata stored without prefix)
        - 'tag.<name>' → looks up column '<name>' (tags stored without prefix)

        Args:
            sample_id: The sample ID to look up

        Returns:
            Dictionary mapping grouping column names to their values

        Raises:
            ValueError: If sample_id not found in compiled_dataset
        """
        if self.compiled_dataset is None:
            raise ValueError("compiled_dataset not initialized")

        # Find any row with this sample_id (all rows for a sample have same grouping values)
        sample_rows = self.compiled_dataset[
            self.compiled_dataset[self._SAMPLE_ID_COLUMN] == sample_id
        ]

        if len(sample_rows) == 0:
            raise ValueError(f"Sample ID '{sample_id}' not found in compiled_dataset")

        # Extract grouping values from first row (they're all the same for this sample)
        first_row = sample_rows.iloc[0]
        grouping_values = {}

        for col in self.grouping_columns:
            # Translate grouping column name to actual DataFrame column name
            actual_col = self._translate_grouping_column(col)

            if actual_col in first_row.index:
                grouping_values[col] = first_row[actual_col]
            else:
                # Column not in dataframe - warn and use None
                trace_message(
                    logger, "OptimalStopping",
                    f"Grouping column '{col}' (mapped to '{actual_col}') not found in compiled_dataset. Using None."
                )
                grouping_values[col] = None

        return grouping_values

    def _translate_grouping_column(self, col: str) -> str:
        """Translate grouping column name to actual DataFrame column name.

        Args:
            col: Grouping column name (e.g., 'model', 'metadata.subtask', 'tag.difficulty')

        Returns:
            Actual DataFrame column name (e.g., 'model', 'subtask', 'difficulty')
        """
        if col.startswith('metadata.'):
            return col[len('metadata.'):]
        elif col.startswith('tag.'):
            return col[len('tag.'):]
        return col

    def _build_grouping_name(self, grouping_values: dict[str, Any]) -> str:
        """Build a consistent grouping name from grouping values dictionary.

        Args:
            grouping_values: Dictionary mapping grouping column names to their values

        Returns:
            String representation of grouping (e.g., "gpt_4-math_task")

        Raises:
            ValueError: If any grouping value contains the reserved delimiter ':::'
        """
        # Validate that no grouping value contains the reserved delimiter
        # The ':::' delimiter is used internally to separate grouping_name from sample_id
        # in stop_sample_ids (format: "grouping_name:::sample_id")
        for col, value in grouping_values.items():
            str_value = str(value) if value is not None else 'None'
            if ':::' in str_value:
                raise ValueError(
                    f"Grouping value for column '{col}' contains reserved delimiter ':::'. "
                    f"Value: '{str_value}'. Please use a different model/task name or metadata value."
                )
        return '-'.join(str(v) if v is not None else 'None' for v in grouping_values.values())

    def _extract_score_value(self, scores: dict[str, SampleScore]) -> float | None:
        """Extract numeric score from scores dictionary with configurable selection/aggregation.

        Supports three modes based on initialization parameters:
        1. Default (both None): Take first score from dict
        2. score_choice: Select specific scorer by name
        3. score_agg: Aggregate all scores using specified method

        If score_value_key is set, dict-valued Score.value objects are
        resolved to the specified key before conversion in all modes.

        Uses inspect_ai's value_to_float() for type conversion if needed.
        The converter is cached at initialization to avoid repeated import attempts.

        Args:
            scores: Dictionary of scorer_name -> SampleScore

        Returns:
            float | None: Extracted/aggregated score, or None if unavailable
        """
        if not scores:
            logger.info("Empty scores dictionary provided")
            return None

        # Use cached converter (initialized in __init__ via _init_value_converter)
        converter = self._value_converter

        def convert_to_float(value: Any) -> float | None:
            """Convert a value to float, handling strings and other types."""
            # Extract from dict if score_value_key is configured
            if isinstance(value, dict):
                if self.score_value_key is not None:
                    if self.score_value_key in value:
                        value = value[self.score_value_key]
                    else:
                        logger.warning(
                            f"score_value_key '{self.score_value_key}' not found in "
                            f"Score.value dict. Available keys: {list(value.keys())}"
                        )
                        return None
                else:
                    logger.warning(
                        f"Score.value is a dict but no score_value_key specified. "
                        f"Available keys: {list(value.keys())}. "
                        f"Set score_value_key to extract the intended numeric value."
                    )
                    return None

            # Guard against other non-scalar types (e.g. lists)
            if isinstance(value, list):
                logger.warning(
                    f"Score.value is a list (length {len(value)}). "
                    f"Expected a scalar value (float, int, or str). "
                    f"Ensure your scorer returns a scalar, not a list."
                )
                return None

            # If already a float or int, return it
            if isinstance(value, (float, int)):
                return float(value)

            # If string, use converter
            if isinstance(value, str):
                if converter is not None:
                    try:
                        return converter(value)
                    except Exception as e:
                        logger.info(f"Converter failed on '{value}': {e}")
                        return None
                else:
                    # Fallback: try direct float conversion
                    try:
                        return float(value)
                    except ValueError:
                        logger.info(f"Could not convert string '{value}' to float")
                        return None

            # Try using converter for other types (booleans, etc.)
            if converter is not None:
                try:
                    return converter(value)
                except Exception:
                    pass

            # Last resort: try as_float() method if available
            if hasattr(value, 'as_float'):
                try:
                    return value.as_float()
                except Exception:
                    pass

            return None

        # Mode 1: Extract specific score by key
        if self.score_choice is not None:
            if self.score_choice not in scores:
                trace_message(
                    logger, "OptimalStopping",
                    f"Requested score key '{self.score_choice}' not found. Available: {list(scores.keys())}"
                )
                return None

            sample_score = scores[self.score_choice]
            value = convert_to_float(sample_score.score.value)

            if value is None or pd.isna(value):
                logger.info(f"Could not convert score '{self.score_choice}' to float")
                return None

            return float(value)

        # Mode 2 & 3: Collect all scores (for aggregation or taking first)
        numeric_scores = []
        for scorer_name, sample_score in scores.items():
            value = convert_to_float(sample_score.score.value)

            if value is not None and not pd.isna(value):
                numeric_scores.append(value)
            else:
                logger.info(f"Could not convert score from scorer '{scorer_name}'")

        if not numeric_scores:
            logger.info("No valid numeric scores found")
            return None

        # Mode 2: Aggregate multiple scores
        if self.score_agg is not None:
            if self.score_agg == 'mean':
                return float(np.mean(numeric_scores))
            elif self.score_agg == 'median':
                return float(np.median(numeric_scores))
            elif self.score_agg == 'mode':
                score_counts = Counter(numeric_scores)
                mode_value = score_counts.most_common(1)[0][0]
                return float(mode_value)
            elif self.score_agg == 'max':
                return float(max(numeric_scores))

        # Mode 3 (Default): Take first score
        return float(numeric_scores[0])

    @override
    async def start_task(self, task: EvalSpec, samples: list[Sample], epochs: int) -> str:
        """Initialize compiled_dataset with full evaluation plan.

        Creates DataFrame with rows for every planned trial:
        - One row per (sample × epoch) combination

        Columns:
        - EvalSpec core columns (model, task, eval_id, etc.)
        - Sample metadata columns (all keys from sample.metadata dictionaries)
        - sample_id (from Sample.id), epoch, score (initially NaN)
        - trial_ran (initially 0)
        - schedule_status (initially True)

        Args:
            task: EvalSpec metadata from inspect_ai
            samples: List of Sample objects with id and metadata
            epochs: Number of epochs per sample

        Returns:
            Name of early stopping manager

        Raises:
            ValueError: If samples list is empty
        """
        # Validate inputs
        if not samples:
            raise ValueError(
                "samples list is empty. Cannot run optimal stopping without samples to evaluate."
            )

        if epochs <= 0:
            raise ValueError(
                f"epochs must be > 0, got {epochs}"
            )

        # 1. Extract EvalSpec core columns (constant across all rows)
        evalspec_columns = {}
        if hasattr(task, 'model') and task.model is not None:
            evalspec_columns['model'] = task.model
        if hasattr(task, 'task') and task.task is not None:
            evalspec_columns['task'] = task.task
        elif hasattr(task, 'task_display_name') and task.task_display_name is not None:
            evalspec_columns['task'] = task.task_display_name
        if hasattr(task, 'eval_id') and task.eval_id is not None:
            evalspec_columns['eval_id'] = task.eval_id

        # 2. Collect all unique metadata keys across all samples
        all_metadata_keys = set()
        for sample in samples:
            if hasattr(sample, 'metadata') and sample.metadata:
                all_metadata_keys.update(sample.metadata.keys())

        # 3. Build cartesian product of all combinations (sample × epoch)
        rows = []
        for sample in samples:
            # Get sample ID
            sample_id = sample.id

            # Get sample metadata (may be incomplete for some samples)
            sample_metadata = {}
            if hasattr(sample, 'metadata') and sample.metadata:
                sample_metadata = sample.metadata

            # Create row for each epoch
            for epoch_num in range(1, epochs + 1):
                row = {
                    # EvalSpec core columns
                    **evalspec_columns,
                    # Sample ID
                    self._SAMPLE_ID_COLUMN: sample_id,
                    # Epoch
                    self._EPOCH_COLUMN: epoch_num,
                    # Sample metadata columns (NaN if missing)
                    **{key: sample_metadata.get(key, np.nan) for key in all_metadata_keys},
                    # Score column (initially empty)
                    self._SCORE_COLUMN: np.nan,
                    # Control columns
                    'trial_ran': 0,
                    'schedule_status': True
                }
                rows.append(row)

        # 4. Create DataFrame
        self.compiled_dataset = pd.DataFrame(rows)

        # 5. Reset tracking variables
        self.stopped_samples = []
        self._decision_counters = {}
        self._stopped_sample_ids = {}
        self._stopped_groupings = set()
        self._stopped_at_trial_count = {}
        self._schedule_cache = {}

        # Reset all PyMC model caches (OPTIMIZATION #2)
        self._binary_item_model_caches = {}
        self._binary_group_model_caches = {}
        self._ordinal_item_model_caches = {}
        self._ordinal_group_model_caches = {}
        self._continuous_item_model_caches = {}
        self._continuous_group_model_caches = {}

        # Reset stabilization and entropy histories for clean state
        self._stabilization_histories = {}
        self._item_entropy_histories = {}

        # Reset pending inference tracking (Fix 7)
        self._pending_inference = {}

        # Recreate executor if it was shutdown (enables manager reuse across evaluations)
        if self._inference_executor._shutdown:
            self._inference_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"optstop_inference_{self.manager_name}"
            )

        # logger.info(
        #     f"Initialized optimal stopping dataset with {len(self.compiled_dataset)} "
        #     f"planned trials ({len(samples)} samples × {epochs} epochs)"
        # )  # Redundant with configuration summary

        # Print configuration summary to console
        self._print_configuration_summary(len(samples), epochs)

        return self.manager_name

    @override
    async def schedule_sample(
        self, id: str | int, epoch: int
    ) -> EarlyStop | None:
        """Check if a sample should be scheduled or stopped early.

        Fast lookup in compiled_dataset to check schedule_status. All necessary
        information is extracted from compiled_dataset (populated during start_task).

        Args:
            id: Sample dataset id
            epoch: Sample epoch

        Returns:
            EarlyStop if the sample should be stopped early, otherwise None

        Note:
            If shadow_mode=True, always returns None (run all trials) without
            checking stopping status. Useful for comparison runs.
        """
        # Shadow mode: run all trials without stopping
        if self.shadow_mode:
            return None

        if self.compiled_dataset is None:
            logger.error("compiled_dataset not initialized. Call start_task() first.")
            return None

        # Get grouping values for this sample from compiled_dataset
        try:
            grouping_values_dict = self._get_grouping_values_for_sample(id)
            grouping_values = tuple(grouping_values_dict.values())
        except ValueError as e:
            logger.error(f"Error getting grouping values: {e}")
            return None

        cache_key = (*grouping_values, id, epoch)

        # Check cache first
        if cache_key in self._schedule_cache:
            should_run = self._schedule_cache[cache_key]
            if not should_run:
                return EarlyStop(
                    id=id,
                    epoch=epoch,
                    reason="Stopped by optimal stopping criteria",
                    metadata={"cache_hit": True}
                )
            return None

        # Build mask for filtering - sample_id + epoch uniquely identifies the row
        # (Grouping columns are already determined by sample_id, so we don't need to filter by them)
        mask = (
            (self.compiled_dataset[self._SAMPLE_ID_COLUMN] == id) &
            (self.compiled_dataset[self._EPOCH_COLUMN] == epoch)
        )

        matching_rows = self.compiled_dataset[mask]

        if len(matching_rows) == 0:
            logger.info(
                f"No matching row found for sample_id={id}, epoch={epoch}"
            )
            return None

        should_run = matching_rows.iloc[0]['schedule_status']

        # Update cache
        self._schedule_cache[cache_key] = should_run

        if not should_run:
            return EarlyStop(
                id=id,
                epoch=epoch,
                reason="Stopped by optimal stopping criteria",
                metadata={"sample_id": id, "epoch": epoch}
            )

        return None

    @override
    async def complete_sample(
        self,
        id: str | int,
        epoch: int,
        scores: dict[str, SampleScore],
    ) -> None:
        """Process completed sample and potentially run optimal stopping inference.

        All necessary information is extracted from compiled_dataset (populated
        during start_task).

        Steps:
        1. Extract and convert score to float
        2. Update compiled_dataset with score and trial_ran=1
        3. Increment decision counter for grouping
        4. If reanalysis_interval reached, run optimal_stopping_live_single()
        5. Update schedule_status based on stopping decisions

        Args:
            id: Sample dataset id
            epoch: Sample epoch
            scores: Scores for this sample
        """
        if self.compiled_dataset is None:
            logger.error("compiled_dataset not initialized. Call start_task() first.")
            return

        # Step 1: Extract score value
        score_value = self._extract_score_value(scores)

        # Step 2: Get grouping values and task name for validation
        try:
            grouping_values = self._get_grouping_values_for_sample(id)
        except ValueError as e:
            logger.error(f"Error getting grouping values: {e}")
            return

        # Extract task name from grouping values for ordinal task checking
        task_name = grouping_values.get('task', None)

        # Step 3: Validate score for inference eligibility
        score_valid_for_inference = True
        validation_message = None

        # Check 1: None or string scores are invalid for inference
        if score_value is None:
            score_valid_for_inference = False
            validation_message = "Score is None - cannot perform inference"
        elif isinstance(score_value, str):
            score_valid_for_inference = False
            validation_message = "Score is a string - cannot perform inference"

        # Check 2: Negative scores are invalid
        elif score_value < 0:
            score_valid_for_inference = False
            validation_message = f"Score is negative ({score_value}) - invalid for inference"

        # Check 3: Score validation depends on aggregation mode and task type
        elif task_name is not None:
            is_aggregated = bool(self.score_agg in ['mean', 'median'])

            # Determine if this is an ordinal task
            is_ordinal = False
            if self.ordinal_tasks:
                # Check if task name matches any ordinal task pattern
                for ordinal_pattern in self.ordinal_tasks:
                    if ordinal_pattern.lower() in str(task_name).lower():
                        is_ordinal = True
                        break

            if not is_ordinal:
                # Binary task validation
                if is_aggregated:
                    # Aggregated binary: allow continuous [0, 1]
                    if score_value > 1.0:
                        score_valid_for_inference = False
                        validation_message = (
                            f"Binary task '{task_name}' with aggregation has score > 1 ({score_value}) - "
                            f"expected continuous values in [0, 1]. Cannot perform inference."
                        )
                else:
                    # Discrete binary: only 0 or 1
                    if score_value > 1:
                        score_valid_for_inference = False
                        validation_message = (
                            f"Binary task '{task_name}' has score > 1 ({score_value}) - "
                            f"expected discrete scores in {{0, 1}}. Cannot perform inference."
                        )

            elif is_ordinal:
                # Ordinal task validation (both discrete and aggregated)
                if score_value > self.ordinal_max_score:
                    score_valid_for_inference = False
                    validation_message = (
                        f"Ordinal task '{task_name}' has score > max ({score_value} > {self.ordinal_max_score}) - "
                        f"exceeds ordinal_max_score. Cannot perform inference."
                    )

        # Log validation result
        if not score_valid_for_inference:
            trace_message(
                logger, "OptimalStopping",
                f"Invalid score for sample_id={id}, epoch={epoch}: {validation_message}. "
                f"Task continues without early stopping for this sample."
            )

        # Step 4: Update compiled_dataset regardless of validation
        # (Record the score even if invalid for inference)
        mask = (
            (self.compiled_dataset[self._SAMPLE_ID_COLUMN] == id) &
            (self.compiled_dataset[self._EPOCH_COLUMN] == epoch)
        )

        # Update the row
        self.compiled_dataset.loc[mask, self._SCORE_COLUMN] = score_value
        self.compiled_dataset.loc[mask, 'trial_ran'] = 1
        self.compiled_dataset.loc[mask, 'schedule_status'] = False  # Already ran

        # Step 5: If score invalid, skip inference for this sample
        if not score_valid_for_inference:
            return

        # Step 6: Increment per-grouping counter (only for valid scores)
        grouping_name = self._build_grouping_name(grouping_values)
        if grouping_name not in self._decision_counters:
            self._decision_counters[grouping_name] = 0
        self._decision_counters[grouping_name] += 1

        # logger.debug(
        #     f"Completed sample_id={id}, epoch={epoch}, score={score_value}. "
        #     f"Grouping '{grouping_name}' counter: {self._decision_counters[grouping_name]}"
        # )  # Verbose per-sample logging

        # Step 7: Check if grouping has already stopped (OPTIMIZATION #1)
        if grouping_name in self._stopped_groupings:
            # logger.debug(
            #     f"Skipping inference for '{grouping_name}' - grouping already stopped"
            # )  # Verbose skip logging
            return

        # Step 8: Check if we should run inference for this grouping
        if self._decision_counters[grouping_name] % self.reanalysis_interval != 0:
            return

        # Step 9: Run optimal stopping inference for this grouping
        await self._run_stopping_inference(grouping_values)

    async def _run_stopping_inference(
        self,
        grouping_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Run optimal_stopping_live_single() and update schedule_status.

        Group-level stopping check runs automatically at the end of sample checks.

        Args:
            grouping_values: Dictionary of grouping column values

        Returns:
            Result dictionary from optimal_stopping_live_single()
        """
        # Build grouping name using helper method for consistency
        grouping_name = self._build_grouping_name(grouping_values)

        # Filter to current grouping
        mask = pd.Series([True] * len(self.compiled_dataset))
        for col, val in grouping_values.items():
            # Translate grouping column name to actual DataFrame column name
            actual_col = self._translate_grouping_column(col)
            if pd.isna(val):
                mask &= self.compiled_dataset[actual_col].isna()
            else:
                mask &= (self.compiled_dataset[actual_col] == val)

        grouping_data = self.compiled_dataset[mask]

        # Only analyze trials that have been run
        completed_data = grouping_data[grouping_data['trial_ran'] == 1].copy()

        # Check minimum samples threshold
        n_completed = len(completed_data[self._SAMPLE_ID_COLUMN].unique())
        if n_completed < self.min_samples_per_grouping:
            # logger.debug(
            #     f"Skipping inference for '{grouping_name}': only {n_completed} completed samples, "
            #     f"minimum is {self.min_samples_per_grouping}"
            # )  # Verbose threshold logging
            return {
                'grouping': grouping_name,
                'stop_sample_ids': [],
                'stop_this_grouping': [],
                'stabilization_history': {},
                'metadata': {}
            }

        logger.info(
            f"Running optimal stopping inference on {len(completed_data)} completed trials "
            f"({n_completed} samples) for '{grouping_name}'"
        )

        # Check for ordinal category sparsity (bridge path - validate_ordinal_scores
        # is not called here, so we need an explicit check)
        is_aggregated = self.score_agg in ['mean', 'median']
        if self.ordinal_tasks and not is_aggregated:
            grouping_name_lower = grouping_name.lower()
            for task in self.ordinal_tasks:
                if task.lower() in grouping_name_lower:
                    from .ordinal_utils import check_ordinal_sparsity
                    scores = completed_data[self._SCORE_COLUMN].dropna().values
                    check_ordinal_sparsity(
                        scores, self.ordinal_max_score, grouping_name
                    )
                    break

        # Import the new function
        from .rule import optimal_stopping_live_single

        # Get or initialize stabilization history for this grouping
        if not hasattr(self, '_stabilization_histories'):
            self._stabilization_histories = {}

        stabilization_history = self._stabilization_histories.get(grouping_name, None)

        # Get or initialize all PyMC model caches for this grouping (OPTIMIZATION #2)
        # Separate caches for item-level and group-level models across all pathways
        if not hasattr(self, '_binary_item_model_caches'):
            self._binary_item_model_caches = {}
        if not hasattr(self, '_binary_group_model_caches'):
            self._binary_group_model_caches = {}
        if not hasattr(self, '_ordinal_item_model_caches'):
            self._ordinal_item_model_caches = {}
        if not hasattr(self, '_ordinal_group_model_caches'):
            self._ordinal_group_model_caches = {}
        if not hasattr(self, '_continuous_item_model_caches'):
            self._continuous_item_model_caches = {}
        if not hasattr(self, '_continuous_group_model_caches'):
            self._continuous_group_model_caches = {}

        # Get or initialize item-level entropy histories for ordinal hybrid mode (Issue #6 fix)
        # This enables Pathway 2 (entropy stabilization) at sample level
        if not hasattr(self, '_item_entropy_histories'):
            self._item_entropy_histories = {}

        # Retrieve caches for this grouping
        model_caches = {
            'binary_item': self._binary_item_model_caches.get(grouping_name, {}),
            'binary_group': self._binary_group_model_caches.get(grouping_name, {}),
            'ordinal_item': self._ordinal_item_model_caches.get(grouping_name, {}),
            'ordinal_group': self._ordinal_group_model_caches.get(grouping_name, {}),
            'continuous_item': self._continuous_item_model_caches.get(grouping_name, {}),
            'continuous_group': self._continuous_group_model_caches.get(grouping_name, {})
        }

        # Retrieve item entropy histories for this grouping (Issue #6 fix)
        item_entropy_histories = self._item_entropy_histories.get(grouping_name, None)

        ## MAJOR FLAG: This is where I feed relevant GPU configuration into sampling_kwargs for optimal_stopping_live_single().
        ## We may want to fix this specifically based on inspect_ai runtime environment.

        # Configure sampling kwargs based on available resources
        # Only check GPU if gpu_ids is explicitly provided as a non-empty list
        if self.gpu_ids is not None and len(self.gpu_ids) > 0:
            gpu_available, gpu_backend, _ = gpu_utils.check_gpu_availability()
        else:
            gpu_available, gpu_backend = False, 'cpu'

        sampling_kwargs = gpu_utils.get_sampling_kwargs(
            params=self.optstop_params,
            gpu_available=gpu_available,
            gpu_backend=gpu_backend,
            num_parallel_tasks=1,
            auto_decide=True
        ) if gpu_available else None

        # Determine if scores are aggregated (mean/median)
        is_aggregated = bool(self.score_agg in ['mean', 'median'])

        # Add aggregation context to params for determine_score_type routing
        params_with_aggregation = self.optstop_params.copy()
        params_with_aggregation['is_aggregated'] = is_aggregated

        # Call optimal_stopping_live_single() in a thread (since it's CPU/GPU intensive)
        import asyncio
        try:
            # Get event loop for executor usage
            loop = asyncio.get_running_loop()

            # === Fix 1: Compute max_n_items for pre-allocation ===
            # This enables model pre-allocation to avoid repeated recompilation
            max_n_items = None
            if self._use_preallocation:
                # Count total unique sample_ids in this grouping (observed + not-yet-observed)
                grouping_mask_full = pd.Series([True] * len(self.compiled_dataset))
                for col, val in grouping_values.items():
                    actual_col = self._translate_grouping_column(col)
                    if pd.isna(val):
                        grouping_mask_full &= self.compiled_dataset[actual_col].isna()
                    else:
                        grouping_mask_full &= (self.compiled_dataset[actual_col] == val)
                max_n_items = self.compiled_dataset.loc[grouping_mask_full, self._SAMPLE_ID_COLUMN].nunique()

            # === Fix 7: Skip Redundant Queued Inference ===
            # If there's already a pending inference for this grouping, skip this one.
            # The pending inference will have all the same data plus more (since samples
            # complete sequentially), so this call's result would be strictly dominated.
            # This eliminates the inference backlog that causes 98% hangs.
            if grouping_name in self._pending_inference:
                pending_future = self._pending_inference[grouping_name]
                if not pending_future.done():
                    logger.debug(
                        f"Skipping redundant inference for '{grouping_name}' - "
                        f"inference already pending with more recent data"
                    )
                    # Return empty result - the pending inference will handle stopping decisions
                    return {
                        'grouping': grouping_name,
                        'stop_sample_ids': [],
                        'stop_this_grouping': [],
                        'stabilization_history': stabilization_history or {},
                        'metadata': {'skipped': True, 'reason': 'redundant_inference'}
                    }

            # Run inference in dedicated executor (no fixed timeout)
            # User controls inference time via draws/tune in optstop_params
            # Using functools.partial to pass keyword arguments to executor
            inference_call = partial(
                optimal_stopping_live_single,
                df_grouping=completed_data,
                grouping_name=grouping_name,
                params=params_with_aggregation,  # Pass updated params with aggregation flag
                sample_id_column=self._SAMPLE_ID_COLUMN,
                epoch_column=self._EPOCH_COLUMN,
                score_column=self._SCORE_COLUMN,
                stabilization_history=stabilization_history,
                max_n_items=max_n_items,  # Fix 1: Enable model pre-allocation
                ordinal_tasks=self.ordinal_tasks,
                ordinal_max_score=self.ordinal_max_score,
                ordinal_inference=self.ordinal_inference,
                ordinal_model_type=self.ordinal_model_type,
                entropy_threshold=self.entropy_threshold,
                prior_mu=self.prior_mu,
                prior_sigma=self.prior_sigma,
                sampling_kwargs=sampling_kwargs,
                model_caches=model_caches,  # OPTIMIZATION #2: Persist all PyMC models
                item_entropy_histories=item_entropy_histories  # Issue #6 fix: Persist for Pathway 2
            )

            # Submit inference and track the Future (Fix 7)
            future = loop.run_in_executor(
                self._inference_executor,
                inference_call
            )
            self._pending_inference[grouping_name] = future

            try:
                result = await future
            finally:
                # Clean up tracking - only remove if this is still the tracked future
                # (another call may have replaced it)
                if self._pending_inference.get(grouping_name) is future:
                    del self._pending_inference[grouping_name]
        except asyncio.CancelledError:
            trace_message(
                logger, "OptimalStopping",
                f"Inference cancelled for '{grouping_name}'. Evaluation may have been interrupted."
            )
            # Return safe default - no stopping decisions
            return {
                'stopped_samples': [],
                'stopped_groupings': [],
                'metadata': {
                    'error': 'cancelled',
                    'grouping': grouping_name
                }
            }
        except Exception as e:
            trace_message(
                logger, "OptimalStopping",
                f"Error running inference for '{grouping_name}': {e}"
            )
            # Return safe default - no stopping decisions
            return {
                'grouping': grouping_name,
                'stop_sample_ids': [],
                'stop_this_grouping': [],
                'stabilization_history': stabilization_history or {},
                'metadata': {'error': str(e)}
            }

        # Update stored stabilization history
        self._stabilization_histories[grouping_name] = result['stabilization_history']

        # Update stored model caches (OPTIMIZATION #2)
        if 'model_caches' in result:
            returned_caches = result['model_caches']
            self._binary_item_model_caches[grouping_name] = returned_caches.get('binary_item', {})
            self._binary_group_model_caches[grouping_name] = returned_caches.get('binary_group', {})
            self._ordinal_item_model_caches[grouping_name] = returned_caches.get('ordinal_item', {})
            self._ordinal_group_model_caches[grouping_name] = returned_caches.get('ordinal_group', {})
            self._continuous_item_model_caches[grouping_name] = returned_caches.get('continuous_item', {})
            self._continuous_group_model_caches[grouping_name] = returned_caches.get('continuous_group', {})

        # Update stored item entropy histories for ordinal hybrid mode (Issue #6 fix)
        if 'item_entropy_histories' in result:
            self._item_entropy_histories[grouping_name] = result['item_entropy_histories']

        # Initialize stopped sample tracking for this grouping if needed
        if grouping_name not in self._stopped_sample_ids:
            self._stopped_sample_ids[grouping_name] = set()

        # Process stopped sample_ids
        for stop_id_str in result['stop_sample_ids']:
            # Format is "grouping_name:::sample_id" (delimiter: :::)
            # Extract just the sample_id part
            parts = stop_id_str.split(':::', 1)
            if len(parts) != 2:
                trace_message(
                    logger, "OptimalStopping",
                    f"Unexpected format for stop_id_str: '{stop_id_str}'. Expected: 'grouping_name:::sample_id'"
                )
                continue
            sample_id = parts[1]

            # Skip if we've already processed this sample's stopping decision
            if sample_id in self._stopped_sample_ids[grouping_name]:
                continue

            # Mark as stopped to prevent future duplicates
            self._stopped_sample_ids[grouping_name].add(sample_id)

            # Update compiled_dataset: set schedule_status=False for remaining epochs
            update_mask = mask & (self.compiled_dataset[self._SAMPLE_ID_COLUMN] == sample_id)
            update_mask &= (self.compiled_dataset['trial_ran'] == 0)  # Only unrun trials

            self.compiled_dataset.loc[update_mask, 'schedule_status'] = False

            # Clear cache for affected epochs (optimized with .unique())
            affected_epochs = self.compiled_dataset.loc[update_mask, self._EPOCH_COLUMN].unique()
            grouping_tuple = tuple(grouping_values.values())
            for epoch in affected_epochs:
                self._schedule_cache.pop((*grouping_tuple, sample_id, epoch), None)

            # Record stopped sample (only first time)
            if result['metadata'].get('sample_stopping_reasons', {}).get(str(sample_id)):
                reason_info = result['metadata']['sample_stopping_reasons'][str(sample_id)]
                stopped_epoch = reason_info.get('epochs_used', 0)
                self.stopped_samples.append(StoppedSample(
                    id=sample_id,
                    epoch=stopped_epoch,
                    early_stop=EarlyStop(
                        id=sample_id,
                        epoch=stopped_epoch,
                        reason=reason_info.get('reason', 'optimal_stopping'),
                        metadata=reason_info
                    )
                ))

                # Sample stopping details available in complete_task() diagnostics
                # reason = reason_info.get('reason', 'unknown')
                # epochs_used = reason_info.get('epochs_used', 0)
                # ... (logging commented out - info in stopped_samples output)
            # else:
            #     logger.info(f"Marked sample {sample_id} for early stopping in grouping '{grouping_name}'")

        # Process group-level stopping
        if result['stop_this_grouping']:
            # Skip if we've already stopped this grouping
            if grouping_name not in self._stopped_groupings:
                # Mark as stopped to prevent future duplicates
                self._stopped_groupings.add(grouping_name)

                # Record when stopping was first triggered
                self._stopped_at_trial_count[grouping_name] = {
                    'global_trial_count': int(self.compiled_dataset['trial_ran'].sum()),
                    'grouping_completed_samples': self._decision_counters.get(grouping_name, 0),
                }

                # Set schedule_status=False for ALL remaining unrun trials in this grouping
                update_mask = mask & (self.compiled_dataset['trial_ran'] == 0)
                self.compiled_dataset.loc[update_mask, 'schedule_status'] = False

                # Clear cache for entire grouping (optimized with dict comprehension)
                grouping_prefix = tuple(grouping_values.values())
                self._schedule_cache = {
                    k: v for k, v in self._schedule_cache.items()
                    if k[:len(grouping_prefix)] != grouping_prefix
                }

                # Grouping stopping details available in complete_task() diagnostics
                # (Detailed logging commented out - info in stopped_groupings output)

        return result

    def _build_stabilization_entry(self, grouping_name: str, history: dict) -> dict:
        """Build stabilization history entry for a grouping.

        Only includes ordinal-specific fields when the grouping uses ordinal inference.

        Args:
            grouping_name: Name of the grouping (e.g., 'gpt-4-turbo-math_easy')
            history: Raw stabilization history dict for this grouping

        Returns:
            Cleaned stabilization entry with appropriate fields for the score type
        """
        # Base fields for all score types
        entry = {
            'n_samples': history.get('n_samples_evaluated', 0),
            'final_ci_width': history['ci_width_history'][-1] if history.get('ci_width_history') else None,
            'final_slope': history['ci_slope_history'][-1] if history.get('ci_slope_history') else None,
            'n_group_checks': len(history.get('ci_width_history', [])),
        }

        # Convergence projection for non-stopped groupings
        if grouping_name not in self._stopped_groupings and entry['final_ci_width'] is not None:
            from .convergence import project_convergence
            # Apply conservatism-adjusted slope_threshold when performance is low,
            # matching the logic in rule.py (CI_delta / conservatism for low-perf).
            ci_delta = self.optstop_params.get('CI_delta', 0.00001)
            conservatism = self.optstop_params.get('conservatism', 5)
            low_perf_threshold = self.optstop_params.get('low_performance_threshold', 0.01)
            current_perf = history.get('current_perf_estimate', 0.0)
            if current_perf < low_perf_threshold:
                adjusted_slope_threshold = ci_delta / conservatism
            else:
                adjusted_slope_threshold = ci_delta
            projection = project_convergence(
                ci_widths=history.get('ci_width_history', []),
                ci_slopes=history.get('ci_slope_history', []),
                delta=self.optstop_params.get('delta_cap', 0.05),
                slope_threshold=adjusted_slope_threshold,
                step_size=self.reanalysis_interval,
                stab_window=self.optstop_params.get('stab_window', 15),
            )
            if projection is not None:
                entry['convergence_projection'] = projection

        # Only include ordinal-specific fields if this grouping uses DISCRETE ordinal inference
        # A grouping uses discrete ordinal if:
        # 1. ordinal_tasks is set AND the task matches, AND
        # 2. score_agg is NOT used (aggregation reroutes to continuous_bounded inference)
        # This matches the routing logic in ordinal_utils.determine_score_type()
        is_ordinal_grouping = False
        is_aggregated = self.score_agg in ['mean', 'median']
        if self.ordinal_tasks and not is_aggregated:
            # Use case-insensitive matching to match routing logic in ordinal_utils.determine_score_type()
            grouping_name_lower = grouping_name.lower()
            for task in self.ordinal_tasks:
                if task.lower() in grouping_name_lower:
                    is_ordinal_grouping = True
                    break

        if is_ordinal_grouping:
            # Add ordinal-specific fields
            entry.update({
                'final_modal_ci_width': history.get('final_modal_ci_width'),
                'final_modal_ci': history.get('final_modal_ci'),
                'final_entropy': history.get('final_entropy'),
                'final_entropy_threshold': history.get('final_entropy_threshold'),
                'final_entropy_ci_width': history.get('final_entropy_ci_width'),
                'final_convergence_threshold': history.get('final_convergence_threshold'),
                'ordinal_pathway': history.get('ordinal_pathway'),
            })

        return entry

    async def complete_task(self) -> dict[str, JsonValue]:
        """Generate final diagnostics and metadata for completed task.

        All necessary information is extracted from compiled_dataset (populated
        during start_task). Group-level stopping checks run automatically during
        _run_stopping_inference() calls, so no additional check is needed here.

        In shadow mode, a shadow_mode_summary block is appended with
        would_have_stopped_at (earliest global trial count across groupings)
        and potential_efficiency_percent. Per-grouping details are in
        stopped_at_trial_count.

        Returns:
            Metadata dictionary with diagnostics, stopping decisions, and efficiency stats
        """
        if self.compiled_dataset is None:
            return {"error": "compiled_dataset not initialized"}

        # Shutdown inference executor gracefully
        # wait=True ensures any running inference completes before proceeding
        # This blocks until PyMC worker pools terminate cleanly
        # logger.info("Shutting down inference executor...")  # Verbose shutdown logging
        self._inference_executor.shutdown(wait=True)
        # logger.info("Inference executor shutdown complete.")  # Verbose shutdown logging

        # Calculate summary statistics
        total_planned = len(self.compiled_dataset)
        total_ran = int(self.compiled_dataset['trial_ran'].sum())
        total_skipped = total_planned - total_ran
        efficiency = (total_skipped / total_planned * 100) if total_planned > 0 else 0

        ## MAJOR FLAG: Confirm this is the desired format for final metadata output.

        # Build metadata with comprehensive diagnostics
        metadata = {
            "manager": self.manager_name,
            "random_seed": self.random_seed,
            "seed_source": self._seed_source,
            "total_planned_trials": total_planned,
            "total_ran": total_ran,
            "total_skipped": total_skipped,
            "efficiency_percent": round(efficiency, 2),
            "stopped_samples_count": len(self.stopped_samples),
            "stopped_samples": [
                {
                    "id": str(sample.id),
                    "epoch": sample.epoch,
                    "reason": sample.early_stop.reason,
                    "metadata": sample.early_stop.metadata
                }
                for sample in self.stopped_samples
            ],
            "grouping_columns": self.grouping_columns,
            "reanalysis_interval": self.reanalysis_interval,
            "min_samples_per_grouping": self.min_samples_per_grouping,
            "stopped_samples_per_grouping": {
                grouping: len(sample_ids)
                for grouping, sample_ids in self._stopped_sample_ids.items()
            },
            "stopped_groupings": list(self._stopped_groupings),
            "stopped_groupings_count": len(self._stopped_groupings),
            "stopped_at_trial_count": self._stopped_at_trial_count,
            "decision_counters": {
                grouping: {
                    'completed_samples': count,
                    'inference_calls': count // self.reanalysis_interval,
                    'next_inference_at': (count // self.reanalysis_interval + 1) * self.reanalysis_interval
                }
                for grouping, count in self._decision_counters.items()
            },
            "stabilization_histories": {
                k: self._build_stabilization_entry(k, v)
                for k, v in self._stabilization_histories.items()
            }
        }

        # Shadow mode: compute potential efficiency from recorded stopping points
        if self.shadow_mode and self._stopped_at_trial_count:
            stop_trials = {
                g: info['global_trial_count']
                for g, info in self._stopped_at_trial_count.items()
            }
            first_stop = min(stop_trials.values())
            metadata["shadow_mode_summary"] = {
                "stopped_at_trial_count": stop_trials,
                "would_have_stopped_at": first_stop,
                "potential_efficiency_percent": round(
                    (1 - first_stop / total_planned) * 100, 2
                ) if total_planned > 0 else 0.0,
            }

        # Add glossary for ordinal stopping reasons
        # Only include if using discrete ordinal inference (not aggregated to continuous)
        is_aggregated = self.score_agg in ['mean', 'median']
        if self.ordinal_tasks and not is_aggregated:
            metadata['ordinal_glossary'] = {
                'modal_ci_narrow_validated': {
                    'description': 'Bootstrap modal confidence interval was narrow and validated by low entropy',
                    'interpretation': 'Distribution is peaked (most responses in same category) with high certainty',
                    'metrics': 'modal_width < threshold AND entropy < entropy_threshold'
                },
                'entropy_converged': {
                    'description': 'Entropy CI width narrow enough for precise distribution estimate',
                    'interpretation': 'Entropy known to within ±5% of max (width < 0.10 on [0,1] scale)',
                    'metrics': 'entropy_width_scaled < entropy_convergence_threshold (default 0.10)'
                },
                'entropy_converged_hierarchical': {
                    'description': 'Hierarchical entropy CI width narrow enough for precise distribution estimate',
                    'interpretation': 'Entropy known to within ±5% of max (width < 0.10 on [0,1] scale)',
                    'metrics': 'entropy_width < entropy_convergence_threshold (default 0.10)'
                },
                'continue_insufficient_history': {
                    'description': 'Need more entropy checks before Pathway 2 can fire',
                    'interpretation': 'Collecting more data for reliable entropy CI estimates',
                    'metrics': 'epochs_tracked < min_epochs_for_stabilization (default 3)'
                }
            }

        # logger.info(
        #     f"Task complete. Ran {total_ran}/{total_planned} trials "
        #     f"({efficiency:.1f}% efficiency gain)"
        # )  # Summary available in returned metadata

        return metadata