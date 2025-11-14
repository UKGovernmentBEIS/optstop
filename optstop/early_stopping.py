from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Any, Optional
import pandas as pd
import numpy as np
import logging
from collections import Counter

from pydantic import BaseModel, Field, JsonValue

# Import GPU utilities for configuration
from . import gpu_utils

if TYPE_CHECKING:
    from inspect_ai.log._log import EvalSpec
    from inspect_ai.scorer._metric import SampleScore

# Configure logger
logger = logging.getLogger(__name__)


class EarlyStop(BaseModel):
    """Directive to stop a sample early."""

    reason: str | None = Field(default=None)
    """Reason for the early stop."""

    metadata: dict[str, JsonValue] | None = Field(default=None)
    """Metadata related to early stop."""


class StoppedSample(BaseModel):
    """Record of early stop for a sample/epoch."""

    id: str | int  ## MINOR FLAG: Consider changing to sample_id for consistency (quick double check with inspect_ai side)
    """Sample dataset id."""

    epoch: int
    """Sample epoch."""

    early_stop: EarlyStop
    """Early stop directive."""

## MINOR FLAG: This is currently redundant, so could be removed unless deemed necessary/useful.
# class EarlyStoppingSummary(BaseModel):
#     manager: str
#     """Name of early stopping manager."""

#     stopped_samples: list[StoppedSample]
#     """Samples that were stopped early."""

#     metadata: dict[str, JsonValue]
#     """Metadata about early stopping"""


class EarlyStopping(Protocol):
    compiled_dataset: Optional[pd.DataFrame]

    async def start_task(self, task: "EvalSpec") -> str:
        """Called at the beginning of an eval run to initialize the stopping manager.

        Implementations should build an internal dataset with all planned trials
        (every grouping × sample_id × epoch combination) and initialize any
        tracking structures needed for incremental stopping decisions.

        Args:
            task: Task metadata from inspect_ai containing dataset, config, etc.

        Returns:
            Name of early stopping manager (for logging/identification).

        Raises:
            ValueError: If task structure is invalid (e.g., missing dataset/samples).
        """
        ...

    async def schedule_sample(
        self, task: "EvalSpec", id: str | int, epoch: int
    ) -> EarlyStop | None:
        """Called before scheduling a sample to check for early stop directive.

        Implementations should perform a fast lookup (typically O(1) with caching)
        to determine if this specific trial should be skipped due to a previous
        stopping decision.

        Args:
            task: Task metadata for extracting grouping values.
            id: Sample dataset id (matches task.dataset.sample_ids).
            epoch: Sample epoch number (1-indexed).

        Returns:
            EarlyStop with reason/metadata if trial should be skipped, None if should run.
        """
        ...

    async def complete_sample(
        self,
        task: "EvalSpec",
        id: str | int,
        epoch: int,
        scores: dict[str, "SampleScore"],
    ) -> None:
        """Called when a sample trial completes with results.

        Implementations should:
        1. Extract and validate the score value
        2. Update internal tracking with completed trial
        3. Periodically run optimal stopping inference (based on reanalysis_interval)
        4. Update stopping decisions and mark samples/groupings as stopped if criteria met

        This is where the core optimal stopping logic runs incrementally as trials complete.

        Args:
           task: Task metadata for extracting grouping values.
           id: Sample dataset id that completed.
           epoch: Sample epoch that completed.
           scores: Dictionary of scores from inspect_ai scorers (may have multiple).
        """
        ...

    async def complete_task(self, task: "EvalSpec") -> dict[str, JsonValue]:
        """Called when the entire evaluation task completes.

        Implementations should return comprehensive diagnostics including:
        - Total trials planned vs. actually run (efficiency metrics)
        - Per-grouping stopping decisions and reasons
        - Stopped samples with metadata
        - Inference timing statistics
        - Any error/warning information

        Args:
           task: Task metadata (may not be needed, but provided for consistency).

        Returns:
            Dictionary of metadata and diagnostics for logging/analysis.
        """
        ...


class OptimalStoppingManager:
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

    def __init__(
        self,
        optstop_params: dict[str, Any],
        grouping_columns: list[str],
        score_column: str = "score",
        sample_id_column: str = "sample_id",
        epoch_column: str = "epoch",
        reanalysis_interval: int = 10,
        min_samples_per_grouping: int = 5,
        ordinal_tasks: Optional[list[str]] = None,
        ordinal_max_score: int = 10,
        ordinal_inference: str = 'hybrid',
        gpu_ids: Optional[list[int]] = None,
        max_workers: Optional[int] = None,
        manager_name: str = "optstop"
    ):
        """Initialize optimal stopping manager.

        Args:
            optstop_params: Dictionary of optimal stopping parameters
                (delta_item, delta_cap, cred_level, conservatism, etc.)
            grouping_columns: List of columns to use for grouping decisions.
                REQUIRED - user must specify.
                Recommended: ['model', 'task']

                Supported syntax:
                - 'model' -> EvalSpec.model
                - 'task' -> EvalSpec.task
                - 'metadata.<key>' -> EvalSpec.metadata[<key>]
                - 'tag.<name>' -> Check if <name> in EvalSpec.tags

            score_column: Name for score column in compiled_dataset
            sample_id_column: Name for sample_id column
            epoch_column: Name for epoch column
            reanalysis_interval: Run inference every N completed samples
            min_samples_per_grouping: Minimum completed samples before first inference
            ordinal_tasks: Task names/substrings using ordinal scoring
            ordinal_max_score: Maximum ordinal score value
            ordinal_inference: Ordinal inference mode ('modal', 'entropy', 'hybrid')
            gpu_ids: GPU IDs to use for computation
            max_workers: Max parallel workers
            manager_name: Name identifier for this manager
        """
        # Configuration
        self.optstop_params = optstop_params
        self.grouping_columns = grouping_columns
        self.score_column = score_column
        self.sample_id_column = sample_id_column
        self.epoch_column = epoch_column
        self.reanalysis_interval = reanalysis_interval
        self.min_samples_per_grouping = min_samples_per_grouping
        self.manager_name = manager_name

        # Ordinal configuration
        self.ordinal_tasks = ordinal_tasks
        self.ordinal_max_score = ordinal_max_score
        self.ordinal_inference = ordinal_inference

        # GPU configuration
        self.gpu_ids = gpu_ids
        self.max_workers = max_workers

        # Data tracking
        self.compiled_dataset: Optional[pd.DataFrame] = None
        self.stopped_samples: list[StoppedSample] = []

        # Per-grouping decision counters for consistent inference timing
        self._decision_counters: dict[str, int] = {}

        # Track stopped samples per grouping to prevent duplicate logging
        self._stopped_sample_ids: dict[str, set] = {}

        # Track stopped groupings to prevent duplicate logging
        self._stopped_groupings: set[str] = set()

        # Cache for fast lookups
        self._schedule_cache: dict[tuple, bool] = {}

        # Stabilization histories per grouping
        self._stabilization_histories: dict[str, dict[str, list[float]]] = {}

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
        cred_level = self.optstop_params.get('cred_level')  # default: 0.95
        if cred_level is not None and not (0 < cred_level < 1):
            raise ValueError(f"cred_level must be between 0 and 1, got {cred_level}")

        conservatism = self.optstop_params.get('conservatism')  # default: 5
        if conservatism is not None and conservatism < 1:
            raise ValueError(f"conservatism must be >= 1, got {conservatism}")

        low_perf_threshold = self.optstop_params.get('low_performance_threshold')  # default: 0.01
        if low_perf_threshold is not None and not (0 <= low_perf_threshold <= 1):
            raise ValueError(f"low_performance_threshold must be between 0 and 1, got {low_perf_threshold}")

        # Stabilization parameters
        CI_delta = self.optstop_params.get('CI_delta')  # default: 0.00005
        if CI_delta is not None and CI_delta <= 0:
            raise ValueError(f"CI_delta must be > 0, got {CI_delta}")

        stab_window = self.optstop_params.get('stab_window')  # default: 10
        if stab_window is not None and stab_window <= 0:
            raise ValueError(f"stab_window must be > 0, got {stab_window}")

        rep_batch_size = self.optstop_params.get('rep_batch_size')  # default: 1
        if rep_batch_size is not None and rep_batch_size <= 0:
            raise ValueError(f"rep_batch_size must be > 0, got {rep_batch_size}")

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

        # 6. Validate column names are not empty
        if not self.score_column:
            raise ValueError("score_column cannot be empty")
        if not self.sample_id_column:
            raise ValueError("sample_id_column cannot be empty")
        if not self.epoch_column:
            raise ValueError("epoch_column cannot be empty")

    def _extract_grouping_value(self, task: "EvalSpec", column: str) -> Any:
        """Extract value for a grouping column from EvalSpec.

        Supports:
        - 'model' -> task.model
        - 'task' -> task.task (or task_display_name if task is None)
        - 'metadata.<key>' -> task.metadata[<key>]
        - 'tag.<name>' -> '<name>' if in task.tags else None

        Args:
            task: EvalSpec instance
            column: Grouping column specification

        Returns:
            Extracted value or None if not found

        Raises:
            ValueError: If column specification is invalid
        """
        if column == 'model':
            return task.model
        elif column == 'task':
            return task.task or task.task_display_name
        elif column.startswith('metadata.'):
            key = column.split('.', 1)[1]
            if task.metadata and key in task.metadata:
                return task.metadata[key]
            else:
                # MINOR FLAG: Missing metadata key - warn user
                logger.warning(
                    f"Grouping column '{column}' not found in task.metadata. "
                    f"Using None as grouping value."
                )
                return None
        elif column.startswith('tag.'):
            tag_name = column.split('.', 1)[1]
            if task.tags and tag_name in task.tags:
                return tag_name
            return None
        else:
            raise ValueError(
                f"Invalid grouping column '{column}'. "
                f"Supported formats: 'model', 'task', 'metadata.<key>', 'tag.<name>'"
            )

    def _build_grouping_name(self, grouping_values: dict[str, Any]) -> str:
        """Build a consistent grouping name from grouping values dictionary.

        Args:
            grouping_values: Dictionary mapping grouping column names to their values

        Returns:
            String representation of grouping (e.g., "gpt_4-math_task")
        """
        return '-'.join(str(v) if v is not None else 'None' for v in grouping_values.values())

    def _extract_score_value(self, scores: dict[str, "SampleScore"]) -> float | None:
        """Extract numeric score from potentially multiple scorers.

        Uses inspect_ai's value_to_float() for conversion of special values
        (CORRECT, INCORRECT, PARTIAL, NOANSWER, booleans, etc.).

        If multiple scorers provide scores, takes the mode. If scores cannot
        be converted to float or are all None/NaN, returns None.

        Args:
            scores: Dictionary of scorer_name -> SampleScore

        Returns:
            float | None: Mode of numeric scores, or None if unavailable
        """

        ## MAJOR FLAG: Confirm appropriate approach to dealing with scoring (e.g., taking correct user inputs for multiple scorers, etc.)

        # Import inspect_ai's conversion function
        try:
            from inspect_ai.scorer._metric import value_to_float
            converter = value_to_float()
        except ImportError:
            logger.warning(
                "Could not import value_to_float from inspect_ai. "
                "Falling back to basic float conversion."
            )
            converter = None

        numeric_scores = []

        for scorer_name, sample_score in scores.items():
            try:
                # First try using inspect_ai's converter if available
                if converter is not None:
                    value = converter(sample_score.score.value)
                else:
                    # Fallback to as_float() method
                    value = sample_score.score.as_float()

                if value is not None and not pd.isna(value):
                    numeric_scores.append(value)
            except (ValueError, TypeError, AttributeError) as e:
                # Non-numeric or missing score
                logger.debug(
                    f"Could not convert score from scorer '{scorer_name}': {e}"
                )
                continue

        if not numeric_scores:
            return None

        if len(numeric_scores) == 1:
            return numeric_scores[0]

        # MAJOR FLAG: Here is the particular area where decisions are being made about multiple scores - take mode
        # Count occurrences
        score_counts = Counter(numeric_scores)
        mode_value, mode_count = score_counts.most_common(1)[0]

        # Check if multimodal (multiple values with same max count)
        max_count = mode_count
        modes = [val for val, count in score_counts.items() if count == max_count]

        if len(modes) > 1:
            logger.debug(
                f"Multiple scores are multimodal: {modes}. "
                f"Taking first mode: {mode_value}"
            )

        return float(mode_value)

    async def start_task(self, task: "EvalSpec") -> str:
        """Initialize compiled_dataset with full evaluation plan.

        Creates DataFrame with rows for every planned trial:
        - One row per (grouping × sample_id × epoch) combination

        Columns:
        - Grouping columns (from user config)
        - sample_id, epoch, score (initially NaN)
        - trial_ran (initially 0)
        - schedule_status (initially True)

        Args:
            task: Task metadata from inspect_ai

        Returns:
            Name of early stopping manager

        Raises:
            ValueError: If task.dataset is None or sample_ids is empty
        """
        ## MAJOR FLAG: Need to confirm the extraction of information from EvalSpec is correct.

        # Validate task structure
        if task.dataset is None:
            raise ValueError(
                "task.dataset is None. Cannot initialize optimal stopping without dataset."
            )

        if task.config is None:
            logger.warning(
                "task.config is None. Defaulting to 1 epoch per sample."
            )
            epochs = 1
        else:
            epochs = task.config.epochs or 1

        # Validate sample_ids
        sample_ids = task.dataset.sample_ids
        if not sample_ids:
            raise ValueError(
                "task.dataset.sample_ids is empty. Cannot run optimal stopping "
                "without samples to evaluate."
            )

        # 1. Extract grouping values for all grouping columns
        grouping_values = {}
        for col in self.grouping_columns:
            grouping_values[col] = self._extract_grouping_value(task, col)

        # 3. Build cartesian product of all combinations
        rows = []
        for sample_id in sample_ids:
            for epoch in range(1, epochs + 1):
                row = {
                    **grouping_values,
                    self.sample_id_column: sample_id,
                    self.epoch_column: epoch,
                    self.score_column: np.nan,
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
        self._schedule_cache = {}

        logger.info(
            f"Initialized optimal stopping dataset with {len(self.compiled_dataset)} "
            f"planned trials ({len(sample_ids)} samples × {epochs} epochs)"
        )

        return self.manager_name

    async def schedule_sample(
        self, task: "EvalSpec", id: str | int, epoch: int
    ) -> EarlyStop | None:
        """Check if a sample should be scheduled or stopped early.

        Fast lookup in compiled_dataset to check schedule_status.

        Args:
            task: Task metadata
            id: Sample dataset id
            epoch: Sample epoch

        Returns:
            EarlyStop if the sample should be stopped early, otherwise None
        """
        if self.compiled_dataset is None:
            logger.error("compiled_dataset not initialized. Call start_task() first.")
            return None

        # Build cache key from grouping values + sample_id + epoch
        grouping_values = tuple(
            self._extract_grouping_value(task, col)
            for col in self.grouping_columns
        )
        cache_key = (*grouping_values, id, epoch)

        # Check cache first
        if cache_key in self._schedule_cache:
            should_run = self._schedule_cache[cache_key]
            if not should_run:
                return EarlyStop(
                    reason="Stopped by optimal stopping criteria",
                    metadata={"cache_hit": True}
                )
            return None

        # Build mask for filtering (more robust than query)
        mask = pd.Series([True] * len(self.compiled_dataset))
        for col in self.grouping_columns:
            col_value = self._extract_grouping_value(task, col)
            if col_value is None:
                mask &= self.compiled_dataset[col].isna()
            else:
                mask &= (self.compiled_dataset[col] == col_value)

        mask &= (self.compiled_dataset[self.sample_id_column] == id)
        mask &= (self.compiled_dataset[self.epoch_column] == epoch)

        matching_rows = self.compiled_dataset[mask]

        if len(matching_rows) == 0:
            logger.warning(
                f"No matching row found for sample_id={id}, epoch={epoch}"
            )
            return None

        should_run = matching_rows.iloc[0]['schedule_status']

        # Update cache
        self._schedule_cache[cache_key] = should_run

        if not should_run:
            return EarlyStop(
                reason="Stopped by optimal stopping criteria",
                metadata={"sample_id": id, "epoch": epoch}
            )

        return None

    async def complete_sample(
        self,
        task: "EvalSpec",
        id: str | int,
        epoch: int,
        scores: dict[str, "SampleScore"],
    ) -> None:
        """Process completed sample and potentially run optimal stopping inference.

        Steps:
        1. Extract and convert score to float
        2. Update compiled_dataset with score and trial_ran=1
        3. Increment decision counter
        4. If reanalysis_interval reached, run optimal_stopping_live()
        5. Update schedule_status based on stopping decisions

        Args:
            task: Task metadata
            id: Sample dataset id
            epoch: Sample epoch
            scores: Scores for this sample
        """
        if self.compiled_dataset is None:
            logger.error("compiled_dataset not initialized. Call start_task() first.")
            return

        ## MAJOR FLAG: This is where scoring decisions have impact on stopping logic.
        # Step 1: Extract score value
        score_value = self._extract_score_value(scores)

        # Validate score value
        ## NOTE: This validation is basic; users may want to customize based on their scoring system - this might entail users not just specifying ordinal tasks, but also what the upper bounds are for that specific autograder.
        if score_value is not None and not pd.isna(score_value):
            # Check for suspicious values
            if score_value < 0:
                logger.warning(
                    f"Negative score ({score_value}) for sample_id={id}, epoch={epoch}. "
                    f"This may indicate a data issue."
                )
            elif score_value > 100:  # Arbitrary upper bound for sanity check
                logger.warning(
                    f"Unusually large score ({score_value}) for sample_id={id}, epoch={epoch}. "
                    f"Ensure this is expected for your scoring system."
                )
        else:
            logger.debug(
                f"Non-numeric score for sample_id={id}, epoch={epoch}. "
                f"Marking as non-actionable."
            )

        # Step 2: Update compiled_dataset
        grouping_values = {
            col: self._extract_grouping_value(task, col)
            for col in self.grouping_columns
        }

        # Build update mask
        mask = pd.Series([True] * len(self.compiled_dataset))
        for col, val in grouping_values.items():
            if val is None:
                mask &= self.compiled_dataset[col].isna()
            else:
                mask &= (self.compiled_dataset[col] == val)

        mask &= (self.compiled_dataset[self.sample_id_column] == id)
        mask &= (self.compiled_dataset[self.epoch_column] == epoch)

        # Update the row
        self.compiled_dataset.loc[mask, self.score_column] = score_value
        self.compiled_dataset.loc[mask, 'trial_ran'] = 1
        self.compiled_dataset.loc[mask, 'schedule_status'] = False  # Already ran

        # Step 3: Increment per-grouping counter
        grouping_name = self._build_grouping_name(grouping_values)
        if grouping_name not in self._decision_counters:
            self._decision_counters[grouping_name] = 0
        self._decision_counters[grouping_name] += 1

        logger.debug(
            f"Completed sample_id={id}, epoch={epoch}, score={score_value}. "
            f"Grouping '{grouping_name}' counter: {self._decision_counters[grouping_name]}"
        )

        # Step 4: Check if we should run inference for this grouping
        if self._decision_counters[grouping_name] % self.reanalysis_interval != 0:
            return

        # Run optimal stopping inference for this grouping
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
            if val is None:
                mask &= self.compiled_dataset[col].isna()
            else:
                mask &= (self.compiled_dataset[col] == val)

        grouping_data = self.compiled_dataset[mask]

        # Only analyze trials that have been run
        completed_data = grouping_data[grouping_data['trial_ran'] == 1].copy()

        # Check minimum samples threshold
        n_completed = len(completed_data[self.sample_id_column].unique())
        if n_completed < self.min_samples_per_grouping:
            logger.debug(
                f"Skipping inference for '{grouping_name}': only {n_completed} completed samples, "
                f"minimum is {self.min_samples_per_grouping}"
            )
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

        # Import the new function
        from .rule import optimal_stopping_live_single

        # Get or initialize stabilization history for this grouping
        if not hasattr(self, '_stabilization_histories'):
            self._stabilization_histories = {}

        stabilization_history = self._stabilization_histories.get(grouping_name, None)


        ## MAJOR FLAG: This is where I feed relevant GPU configuration into sampling_kwargs for optimal_stopping_live_single().
        ## We may want to fix this specifically based on inspect_ai runtime environment.

        # Configure sampling kwargs based on available resources
        # Only check GPU if gpu_ids is explicitly provided as a non-empty list
        if self.gpu_ids is not None and len(self.gpu_ids) > 0:
            gpu_available, gpu_backend, gpu_info = gpu_utils.check_gpu_availability()
        else:
            gpu_available, gpu_backend, gpu_info = False, 'cpu', {}

        sampling_kwargs = gpu_utils.get_sampling_kwargs(
            params=self.optstop_params,
            gpu_available=gpu_available,
            gpu_backend=gpu_backend,
            num_parallel_tasks=1,
            auto_decide=True
        ) if gpu_available else None

        # Call optimal_stopping_live_single() in a thread (since it's CPU/GPU intensive)
        import asyncio
        try:
            result = await asyncio.to_thread(
                optimal_stopping_live_single,
                df_grouping=completed_data,
                grouping_name=grouping_name,
                params=self.optstop_params,
                sample_id_column=self.sample_id_column,
                epoch_column=self.epoch_column,
                score_column=self.score_column,
                stabilization_history=stabilization_history,
                ordinal_tasks=self.ordinal_tasks,
                ordinal_max_score=self.ordinal_max_score,
                ordinal_inference=self.ordinal_inference,
                entropy_threshold=1.5,  # Could be added as init parameter if needed
                sampling_kwargs=sampling_kwargs
            )
        except Exception as e:
            logger.error(
                f"Error running optimal stopping inference for '{grouping_name}': {e}",
                exc_info=True
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

        # Initialize stopped sample tracking for this grouping if needed
        if grouping_name not in self._stopped_sample_ids:
            self._stopped_sample_ids[grouping_name] = set()

        # Process stopped sample_ids
        for stop_id_str in result['stop_sample_ids']:
            # Format is "grouping_name:::sample_id" (delimiter: :::)
            # Extract just the sample_id part
            parts = stop_id_str.split(':::', 1)
            if len(parts) != 2:
                logger.error(
                    f"Unexpected format for stop_id_str: '{stop_id_str}'. "
                    f"Expected format: 'grouping_name:::sample_id'"
                )
                continue
            sample_id = parts[1]

            # Skip if we've already processed this sample's stopping decision
            if sample_id in self._stopped_sample_ids[grouping_name]:
                continue

            # Mark as stopped to prevent future duplicates
            self._stopped_sample_ids[grouping_name].add(sample_id)

            # Update compiled_dataset: set schedule_status=False for remaining epochs
            update_mask = mask & (self.compiled_dataset[self.sample_id_column] == sample_id)
            update_mask &= (self.compiled_dataset['trial_ran'] == 0)  # Only unrun trials

            self.compiled_dataset.loc[update_mask, 'schedule_status'] = False

            # Clear cache for affected epochs (optimized with .unique())
            affected_epochs = self.compiled_dataset.loc[update_mask, self.epoch_column].unique()
            grouping_tuple = tuple(grouping_values.values())
            for epoch in affected_epochs:
                self._schedule_cache.pop((*grouping_tuple, sample_id, epoch), None)

            # Record stopped sample (only first time)
            if result['metadata'].get('sample_stopping_reasons', {}).get(str(sample_id)):
                reason_info = result['metadata']['sample_stopping_reasons'][str(sample_id)]
                self.stopped_samples.append(StoppedSample(
                    id=sample_id,
                    epoch=reason_info.get('epochs_used', 0),
                    early_stop=EarlyStop(
                        reason=reason_info.get('reason', 'optimal_stopping'),
                        metadata=reason_info
                    )
                ))

                # Log with reason and key values
                reason = reason_info.get('reason', 'unknown')
                epochs_used = reason_info.get('epochs_used', 0)

                # Build detailed log message based on reason
                if 'ci_width' in reason_info:
                    ci_width = reason_info['ci_width']
                    threshold = reason_info.get('threshold', 'N/A')
                    logger.info(
                        f"Stopped sample {sample_id} in '{grouping_name}' after {epochs_used} epochs: "
                        f"{reason} (CI width={ci_width:.4f}, threshold={threshold})"
                    )
                else:
                    logger.info(
                        f"Stopped sample {sample_id} in '{grouping_name}' after {epochs_used} epochs: {reason}"
                    )
            else:
                logger.info(f"Marked sample {sample_id} for early stopping in grouping '{grouping_name}'")

        # Process group-level stopping
        if result['stop_this_grouping']:
            # Skip if we've already stopped this grouping
            if grouping_name not in self._stopped_groupings:
                # Mark as stopped to prevent future duplicates
                self._stopped_groupings.add(grouping_name)

                # Set schedule_status=False for ALL remaining unrun trials in this grouping
                update_mask = mask & (self.compiled_dataset['trial_ran'] == 0)
                self.compiled_dataset.loc[update_mask, 'schedule_status'] = False

                # Clear cache for entire grouping (optimized with dict comprehension)
                grouping_prefix = tuple(grouping_values.values())
                self._schedule_cache = {
                    k: v for k, v in self._schedule_cache.items()
                    if k[:len(grouping_prefix)] != grouping_prefix
                }

                # Log with reason and key values
                group_reason_info = result['metadata'].get('group_stopping_reason', {})
                if group_reason_info:
                    reason = group_reason_info.get('reason', 'unknown')
                    samples_used = group_reason_info.get('samples_used', 0)

                    # Build metrics string from available fields
                    metrics = []

                    # CI width metrics (binary and some ordinal)
                    if 'ci_width' in group_reason_info:
                        ci_width = group_reason_info['ci_width']
                        metrics.append(f"CI width={ci_width:.4f}")
                    if 'effective_width' in group_reason_info:
                        effective_width = group_reason_info['effective_width']
                        metrics.append(f"effective width={effective_width:.4f}")
                    if 'threshold' in group_reason_info:
                        threshold = group_reason_info['threshold']
                        metrics.append(f"threshold={threshold}")

                    # Stabilization metrics (binary)
                    if 'slope' in group_reason_info:
                        slope = group_reason_info['slope']
                        metrics.append(f"slope={slope:.6f}")
                    if 'slope_threshold' in group_reason_info:
                        slope_threshold = group_reason_info['slope_threshold']
                        metrics.append(f"slope threshold={slope_threshold:.6f}")

                    # Ordinal-specific diagnostics (hybrid, entropy)
                    if 'diagnostics' in group_reason_info:
                        diag = group_reason_info['diagnostics']
                        if isinstance(diag, dict):
                            if 'entropy' in diag:
                                metrics.append(f"entropy={diag['entropy']:.4f}")
                            if 'modal_prob' in diag:
                                metrics.append(f"modal prob={diag['modal_prob']:.4f}")
                            if 'ci_width' in diag and 'ci_width' not in group_reason_info:
                                # Add from diagnostics if not already in main metadata
                                metrics.append(f"CI width={diag['ci_width']:.4f}")

                    # Format log message
                    metrics_str = ", ".join(metrics) if metrics else "no metrics"
                    logger.info(
                        f"Stopped grouping '{grouping_name}' after {samples_used} samples: "
                        f"{reason} ({metrics_str})"
                    )
                else:
                    logger.info(f"Marked entire grouping '{grouping_name}' for early stopping")

        return result

    async def complete_task(self, task: "EvalSpec") -> dict[str, JsonValue]:
        """Generate final diagnostics and metadata for completed task.

        Note: Group-level stopping checks run automatically during
        _run_stopping_inference() calls, so no additional check is needed here.

        Args:
            task: Task metadata

        Returns:
            Metadata dictionary with diagnostics, stopping decisions, and efficiency stats
        """
        if self.compiled_dataset is None:
            return {"error": "compiled_dataset not initialized"}

        # Calculate summary statistics
        total_planned = len(self.compiled_dataset)
        total_ran = int(self.compiled_dataset['trial_ran'].sum())
        total_skipped = total_planned - total_ran
        efficiency = (total_skipped / total_planned * 100) if total_planned > 0 else 0

        ## MAJOR FLAG: Confirm this is the desired format for final metadata output.

        # Build metadata with comprehensive diagnostics
        metadata = {
            "manager": self.manager_name,
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
            "decision_counters": {
                grouping: {
                    'completed_samples': count,
                    'inference_calls': count // self.reanalysis_interval,
                    'next_inference_at': (count // self.reanalysis_interval + 1) * self.reanalysis_interval
                }
                for grouping, count in self._decision_counters.items()
            },
            "stabilization_histories": {
                k: {
                    'n_samples': v.get('n_samples_evaluated', 0),
                    'final_ci_width': v['ci_width_history'][-1] if v.get('ci_width_history') else None,
                    'final_slope': v['ci_slope_history'][-1] if v.get('ci_slope_history') else None,
                    'n_group_checks': len(v.get('ci_width_history', []))
                }
                for k, v in self._stabilization_histories.items()
            }
        }

        logger.info(
            f"Task complete. Ran {total_ran}/{total_planned} trials "
            f"({efficiency:.1f}% efficiency gain)"
        )

        return metadata