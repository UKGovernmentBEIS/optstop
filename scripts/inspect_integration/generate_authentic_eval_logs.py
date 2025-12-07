#!/usr/bin/env python3
"""
Generate authentic inspect_ai .eval log files with early stopping data.

This script uses the REAL inspect_ai classes from the feature/early-stopping branch
to create properly formatted log files. The data is synthetic (mock LLM responses)
but the format is guaranteed to be correct because we use inspect_ai's own
Pydantic models and serialization.

Prerequisites:
    Run setup_inspect_branch.sh first to create the virtual environment with
    the feature/early-stopping branch installed.

Usage:
    source scripts/inspect_integration/venv_inspect/bin/activate
    PYTHONPATH=/home/ubuntu/optstop:$PYTHONPATH python scripts/inspect_integration/generate_authentic_eval_logs.py

Output:
    - test_outputs/authentic_eval_logs/eval_binary_stopping.eval
    - test_outputs/authentic_eval_logs/eval_binary_stopping.json
    - test_outputs/authentic_eval_logs/eval_continuous_stopping.eval
    - test_outputs/authentic_eval_logs/eval_continuous_stopping.json
    - test_outputs/authentic_eval_logs/eval_ordinal_stopping.eval
    - test_outputs/authentic_eval_logs/eval_ordinal_stopping.json
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np

# Import REAL inspect_ai classes from the feature/early-stopping branch
from inspect_ai.log import (
    EvalLog,
    EvalSample,
    EvalSpec,
    EvalResults,
    EvalStats,
    EvalPlan,
    EvalPlanStep,
    EvalConfig,
    EvalDataset,
    EvalScore,
    EvalMetric,
    write_eval_log,
)
# EvalScorer is internal but needed for scorers field
from inspect_ai.log._log import EvalScorer
from inspect_ai.model import ModelOutput, ModelUsage, ChatMessageUser, ChatMessageAssistant
from inspect_ai.scorer import Score
from inspect_ai.util import EarlyStop, EarlyStoppingSummary

# Import optstop (via PYTHONPATH)
from optstop.early_stopping import OptimalStoppingManager


# =============================================================================
# Mock Score Generator (same as before)
# =============================================================================

class MockScoreGenerator:
    """Generate realistic mock scores."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._sample_base_rates: dict[str, float] = {}

    def _get_sample_base_rate(self, sample_id: str, base_rate: float,
                               sample_variance: float) -> float:
        """Get consistent base rate for a sample."""
        if sample_id not in self._sample_base_rates:
            rate = base_rate + self.rng.uniform(-sample_variance, sample_variance)
            self._sample_base_rates[sample_id] = np.clip(rate, 0.05, 0.95)
        return self._sample_base_rates[sample_id]

    def generate_binary_score(self, sample_id: str, base_rate: float = 0.8,
                               sample_variance: float = 0.15,
                               epoch_variance: float = 0.02) -> float:
        """Generate binary score (0 or 1)."""
        sample_base = self._get_sample_base_rate(sample_id, base_rate, sample_variance)
        epoch_noise = self.rng.uniform(-epoch_variance, epoch_variance)
        final_rate = np.clip(sample_base + epoch_noise, 0.05, 0.95)
        return 1.0 if self.rng.random() < final_rate else 0.0

    def generate_continuous_score(self, sample_id: str, base_rate: float = 0.7,
                                   sample_variance: float = 0.2,
                                   epoch_variance: float = 0.05) -> float:
        """Generate continuous score [0, 1]."""
        sample_base = self._get_sample_base_rate(sample_id, base_rate, sample_variance)
        epoch_noise = self.rng.uniform(-epoch_variance, epoch_variance)
        return float(np.clip(sample_base + epoch_noise, 0.0, 1.0))

    def generate_ordinal_score(self, sample_id: str, max_score: int = 5,
                                base_category: int = 3,
                                sample_variance: float = 1.0,
                                epoch_variance: float = 0.5) -> int:
        """Generate ordinal score [0, max_score]."""
        key = f"ordinal_{sample_id}"
        if key not in self._sample_base_rates:
            rate = base_category + self.rng.uniform(-sample_variance, sample_variance)
            self._sample_base_rates[key] = np.clip(rate, 0, max_score)

        sample_base = self._sample_base_rates[key]
        epoch_noise = self.rng.uniform(-epoch_variance, epoch_variance)
        return int(np.clip(round(sample_base + epoch_noise), 0, max_score))


# =============================================================================
# Create REAL inspect_ai objects
# =============================================================================

def create_eval_sample(
    sample_id: str,
    epoch: int,
    input_text: str,
    target: str,
    output_text: str,
    score_value: float,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> EvalSample:
    """Create a REAL EvalSample using inspect_ai's Pydantic model."""

    # Create real ChatMessage objects
    user_msg = ChatMessageUser(content=input_text, source="input")
    assistant_msg = ChatMessageAssistant(content=output_text, source="generate")

    # Create real ModelOutput
    model_output = ModelOutput.from_content(
        model=model,
        content=output_text,
        stop_reason="stop"
    )

    # Create real Score
    score = Score(value=score_value)

    # Create real ModelUsage
    usage = ModelUsage(input_tokens=100, output_tokens=50, total_tokens=150)

    return EvalSample(
        id=sample_id,
        epoch=epoch,
        input=input_text,
        target=target,
        messages=[user_msg, assistant_msg],
        output=model_output,
        scores={"accuracy": score},
        metadata=metadata or {},
        model_usage={model: usage},
        total_time=1.0,
        working_time=0.8,
    )


def create_eval_spec(
    task_name: str,
    model: str,
    num_samples: int,
    epochs: int,
    created: str,
    metadata: dict[str, Any] | None = None,
) -> EvalSpec:
    """Create a REAL EvalSpec using inspect_ai's Pydantic model."""

    return EvalSpec(
        eval_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        created=created,
        task=task_name,
        task_id=str(uuid.uuid4()),
        task_version=1,
        task_file=f"tasks/{task_name}.py",
        task_display_name=task_name,
        solver="generate",
        tags=["early_stopping_demo", "optstop"],
        dataset=EvalDataset(
            name=f"{task_name}_dataset",
            location=f"datasets/{task_name}.jsonl",
            samples=num_samples,
        ),
        model=model,
        config=EvalConfig(
            epochs=epochs,
            log_samples=True,
        ),
        packages={
            "inspect_ai": "0.3.152-early-stopping",
            "optstop": "0.3.0",
        },
        metadata=metadata,
        scorers=[
            EvalScorer(name="accuracy")
        ],
    )


def create_eval_results(
    total_samples: int,
    completed_samples: int,
    mean_score: float,
    early_stopping_summary: EarlyStoppingSummary,
) -> EvalResults:
    """Create a REAL EvalResults with early stopping data."""

    return EvalResults(
        total_samples=total_samples,
        completed_samples=completed_samples,
        scores=[
            EvalScore(
                name="accuracy",
                scorer="accuracy",
                scored_samples=completed_samples,
                unscored_samples=total_samples - completed_samples,
                metrics={
                    "mean": EvalMetric(name="mean", value=mean_score)
                },
            )
        ],
        early_stopping=early_stopping_summary,
    )


def create_early_stopping_summary(
    manager_name: str,
    diagnostics: dict[str, Any],
    stopped_samples_list: list[dict],
) -> EarlyStoppingSummary:
    """Create a REAL EarlyStoppingSummary from optstop diagnostics."""

    # Convert stopped samples to EarlyStop objects
    early_stops = []
    for sample_info in stopped_samples_list:
        early_stop = EarlyStop(
            id=sample_info["id"],
            epoch=sample_info["epoch"],
            reason=sample_info.get("reason"),
            metadata=sample_info.get("metadata"),
        )
        early_stops.append(early_stop)

    return EarlyStoppingSummary(
        manager=manager_name,
        early_stops=early_stops,
        metadata=diagnostics,
    )


# =============================================================================
# Run Evaluation with Early Stopping
# =============================================================================

async def run_mock_evaluation(
    task_name: str,
    model: str,
    num_samples: int,
    epochs: int,
    score_type: str,
    optstop_params: dict,
    grouping_columns: list[str],
    reanalysis_interval: int = 5,
    seed: int = 42,
    ordinal_max_score: int = 5,
) -> EvalLog:
    """
    Run a mock evaluation with early stopping and return a REAL EvalLog.
    """
    score_generator = MockScoreGenerator(seed=seed)

    # Initialize optstop manager
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=grouping_columns,
        reanalysis_interval=reanalysis_interval,
        min_samples_per_grouping=5,
        ordinal_tasks=['ordinal'] if score_type == 'ordinal' else None,
        ordinal_max_score=ordinal_max_score,
        ordinal_inference='modal' if score_type == 'ordinal' else 'hybrid',
        gpu_ids=None,
        manager_name=f"optstop_{task_name}",
        shadow_mode=False,
        score_agg='mean' if score_type == 'continuous' else None,
        random_seed=seed,
    )

    # Create mock samples for start_task
    mock_samples = []
    for i in range(num_samples):
        sample = Mock()
        sample.id = f"sample_{i:04d}"
        sample.metadata = {"task": task_name}
        mock_samples.append(sample)

    # Create mock EvalSpec for start_task
    mock_eval_spec = Mock()
    mock_eval_spec.model = model
    mock_eval_spec.task = task_name
    mock_eval_spec.eval_id = str(uuid.uuid4())
    mock_eval_spec.metadata = {"score_type": score_type}
    mock_eval_spec.tags = ["early_stopping_demo"]

    # Start task
    created_time = datetime.now(timezone.utc).isoformat()
    await manager.start_task(mock_eval_spec, mock_samples, epochs)

    # Track samples
    eval_samples: list[EvalSample] = []
    stopped_samples_info: list[dict] = []

    # Run evaluation loop
    for epoch in range(1, epochs + 1):
        for sample in mock_samples:
            sample_id = sample.id

            # Check if should schedule
            early_stop = await manager.schedule_sample(sample_id, epoch)

            if early_stop is not None:
                # Record stopped sample info
                stopped_samples_info.append({
                    "id": sample_id,
                    "epoch": epoch,
                    "reason": early_stop.reason,
                    "metadata": early_stop.metadata if hasattr(early_stop, 'metadata') else None,
                })
                continue

            # Generate score based on type
            if score_type == 'binary':
                score = score_generator.generate_binary_score(sample_id)
            elif score_type == 'continuous':
                score = score_generator.generate_continuous_score(sample_id)
            else:
                score = score_generator.generate_ordinal_score(
                    sample_id, max_score=ordinal_max_score
                )

            # Create mock score for complete_sample
            mock_score_obj = Mock()
            mock_score_obj.value = score
            mock_sample_score = Mock()
            mock_sample_score.score = mock_score_obj
            mock_scores = {'accuracy': mock_sample_score}

            await manager.complete_sample(sample_id, epoch, mock_scores)

            # Create REAL EvalSample
            eval_sample = create_eval_sample(
                sample_id=sample_id,
                epoch=epoch,
                input_text=f"Question for {sample_id} (epoch {epoch})",
                target="correct_answer",
                output_text=f"Model response for {sample_id}",
                score_value=float(score),
                model=model,
                metadata={"task": task_name},
            )
            eval_samples.append(eval_sample)

    # Get early stopping diagnostics
    diagnostics = await manager.complete_task()
    completed_time = datetime.now(timezone.utc).isoformat()

    # Calculate statistics
    total_planned = num_samples * epochs
    total_ran = len(eval_samples)
    mean_score = float(np.mean([s.scores["accuracy"].value for s in eval_samples])) if eval_samples else 0.0

    # Create REAL EarlyStoppingSummary
    # Use stopped_samples_info collected from schedule_sample() returns
    # This matches how real inspect_ai collects early_stops from sample_results
    early_stopping_summary = create_early_stopping_summary(
        manager_name=diagnostics.get("manager", f"optstop_{task_name}"),
        diagnostics=diagnostics,
        stopped_samples_list=stopped_samples_info,
    )

    # Create REAL EvalSpec
    eval_spec = create_eval_spec(
        task_name=task_name,
        model=model,
        num_samples=num_samples,
        epochs=epochs,
        created=created_time,
        metadata={"score_type": score_type, "early_stopping_enabled": True},
    )

    # Create REAL EvalResults
    eval_results = create_eval_results(
        total_samples=total_planned,
        completed_samples=total_ran,
        mean_score=mean_score,
        early_stopping_summary=early_stopping_summary,
    )

    # Create REAL EvalStats
    eval_stats = EvalStats(
        started_at=created_time,
        completed_at=completed_time,
        model_usage={model: ModelUsage(input_tokens=10000, output_tokens=5000, total_tokens=15000)},
    )

    # Create REAL EvalPlan
    eval_plan = EvalPlan(
        name="plan",
        steps=[EvalPlanStep(solver="generate", params={})],
    )

    # Create REAL EvalLog
    eval_log = EvalLog(
        version=2,
        status="success",
        eval=eval_spec,
        plan=eval_plan,
        results=eval_results,
        stats=eval_stats,
        samples=eval_samples,
    )

    return eval_log, diagnostics


# =============================================================================
# Main Script
# =============================================================================

async def generate_all_authentic_logs(output_dir: Path, seed: int = 42):
    """Generate all authentic eval log files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model = "anthropic/claude-3-sonnet"
    num_samples = 100
    epochs = 30

    optstop_params = {
        'delta_item': 0.12,          # Tighter sample-level threshold (was 0.15)
        'delta_cap': 0.08,           # Tighter group-level CI threshold (was 0.10)
        'cred_level': 0.95,
        'conservatism': 3,
        'CI_delta': 0.00005,         # Moderate stabilization slope threshold
        'stab_window': 10,           # Standard stabilization window
        'tune': 300,
        'draws': 300,
        'chains': 2,
        'cores': 1,
    }

    print("=" * 80)
    print("Generating AUTHENTIC inspect_ai Eval Logs with Early Stopping")
    print("Using REAL inspect_ai classes from feature/early-stopping branch")
    print("=" * 80)

    # 1. Binary scoring
    print("\n[1/3] Generating binary scoring eval log...")
    binary_log, binary_diag = await run_mock_evaluation(
        task_name="binary_classification",
        model=model,
        num_samples=num_samples,
        epochs=epochs,
        score_type='binary',
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        seed=seed,
    )

    # Write using inspect_ai's REAL write_eval_log function
    binary_eval_path = output_dir / "eval_binary_stopping.eval"
    binary_json_path = output_dir / "eval_binary_stopping.json"

    write_eval_log(binary_log, str(binary_eval_path), format="eval")
    write_eval_log(binary_log, str(binary_json_path), format="json")

    print(f"  Saved: {binary_eval_path}")
    print(f"  Saved: {binary_json_path}")
    print(f"  Samples: {binary_diag['total_ran']}/{binary_diag['total_planned_trials']}")
    print(f"  Efficiency: {binary_diag['efficiency_percent']}%")

    # 2. Continuous scoring
    print("\n[2/3] Generating continuous scoring eval log...")
    continuous_log, continuous_diag = await run_mock_evaluation(
        task_name="continuous_scoring",
        model=model,
        num_samples=num_samples,
        epochs=epochs,
        score_type='continuous',
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        seed=seed + 1,
    )

    continuous_eval_path = output_dir / "eval_continuous_stopping.eval"
    continuous_json_path = output_dir / "eval_continuous_stopping.json"

    write_eval_log(continuous_log, str(continuous_eval_path), format="eval")
    write_eval_log(continuous_log, str(continuous_json_path), format="json")

    print(f"  Saved: {continuous_eval_path}")
    print(f"  Saved: {continuous_json_path}")
    print(f"  Samples: {continuous_diag['total_ran']}/{continuous_diag['total_planned_trials']}")
    print(f"  Efficiency: {continuous_diag['efficiency_percent']}%")

    # 3. Ordinal scoring
    print("\n[3/3] Generating ordinal scoring eval log...")
    ordinal_log, ordinal_diag = await run_mock_evaluation(
        task_name="ordinal_rating",
        model=model,
        num_samples=num_samples,
        epochs=epochs,
        score_type='ordinal',
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        seed=seed + 2,
        ordinal_max_score=5,
    )

    ordinal_eval_path = output_dir / "eval_ordinal_stopping.eval"
    ordinal_json_path = output_dir / "eval_ordinal_stopping.json"

    write_eval_log(ordinal_log, str(ordinal_eval_path), format="eval")
    write_eval_log(ordinal_log, str(ordinal_json_path), format="json")

    print(f"  Saved: {ordinal_eval_path}")
    print(f"  Saved: {ordinal_json_path}")
    print(f"  Samples: {ordinal_diag['total_ran']}/{ordinal_diag['total_planned_trials']}")
    print(f"  Efficiency: {ordinal_diag['efficiency_percent']}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nGenerated 6 eval log files in: {output_dir}")
    print("\nFiles (both .eval binary and .json text formats):")
    print(f"  - eval_binary_stopping.*     ({binary_diag['efficiency_percent']}% efficiency)")
    print(f"  - eval_continuous_stopping.* ({continuous_diag['efficiency_percent']}% efficiency)")
    print(f"  - eval_ordinal_stopping.*    ({ordinal_diag['efficiency_percent']}% efficiency)")

    print("\nEarly Stopping Data Location:")
    print("  log.results.early_stopping -> EarlyStoppingSummary")
    print("    .manager     - Manager name")
    print("    .early_stops - List of EarlyStop objects")
    print("    .metadata    - Full diagnostics from complete_task()")

    print("\nThese files are AUTHENTIC inspect_ai logs created using:")
    print("  - Real Pydantic models from feature/early-stopping branch")
    print("  - Real write_eval_log() serialization")
    print("  - Guaranteed compatibility with Inspect View and log APIs")

    print("\n" + "=" * 80)

    return {
        'binary': (binary_log, binary_diag),
        'continuous': (continuous_log, continuous_diag),
        'ordinal': (ordinal_log, ordinal_diag),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate authentic inspect_ai eval logs with early stopping'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./test_outputs/authentic_eval_logs',
        help='Directory to save output files',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    asyncio.run(generate_all_authentic_logs(output_dir, args.seed))


if __name__ == "__main__":
    main()
