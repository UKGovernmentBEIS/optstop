#!/usr/bin/env python3
"""
Generate mock inspect_ai .json eval log files with early stopping data.

This script creates realistic eval log files that match the inspect_ai EvalLog
format, incorporating early stopping diagnostics from OptimalStoppingManager.

The generated files can be used to demonstrate how early stopping integrates
with the inspect_ai log viewer and evaluation framework.

Usage:
    python scripts/generate_mock_eval_logs.py --output-dir ./test_outputs/mock_eval_logs

Output files:
    - eval_binary_stopping.json      - Binary scoring with early stopping
    - eval_continuous_stopping.json  - Continuous scoring with early stopping
    - eval_ordinal_stopping.json     - Ordinal scoring with early stopping
    - eval_no_stopping.json          - Baseline without early stopping (for comparison)
"""

import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import Mock
import numpy as np

from optstop.early_stopping import OptimalStoppingManager


def make_json_serializable(obj, _seen=None):
    """Recursively convert objects to JSON-serializable types, handling circular refs."""
    # Track seen objects to avoid infinite recursion
    if _seen is None:
        _seen = set()

    # Check for circular reference
    obj_id = id(obj)
    if obj_id in _seen:
        return "<circular_reference>"

    if obj is None:
        return None
    elif isinstance(obj, (str, bool)):
        return obj
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None  # JSON doesn't support NaN/Inf
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [make_json_serializable(x, _seen) for x in obj.tolist()]
    elif isinstance(obj, dict):
        _seen.add(obj_id)
        result = {str(k): make_json_serializable(v, _seen) for k, v in obj.items()}
        _seen.discard(obj_id)
        return result
    elif isinstance(obj, (list, tuple)):
        _seen.add(obj_id)
        result = [make_json_serializable(x, _seen) for x in obj]
        _seen.discard(obj_id)
        return result
    elif hasattr(obj, 'model_dump'):
        # Pydantic models
        _seen.add(obj_id)
        result = make_json_serializable(obj.model_dump(), _seen)
        _seen.discard(obj_id)
        return result
    elif hasattr(obj, '__dict__'):
        # Skip complex objects that could cause cycles (like Mock objects)
        if 'Mock' in type(obj).__name__:
            return f"<Mock:{type(obj).__name__}>"
        _seen.add(obj_id)
        result = make_json_serializable(obj.__dict__, _seen)
        _seen.discard(obj_id)
        return result
    else:
        # Fallback: convert to string
        return str(obj)


# =============================================================================
# inspect_ai Log Structure Definitions (matching main branch)
# =============================================================================

def create_model_usage(input_tokens: int = 100, output_tokens: int = 50) -> dict:
    """Create ModelUsage structure."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_write_tokens": None,
        "cache_read_tokens": None,
        "reasoning_tokens": None
    }


def create_score(value: float, answer: Optional[str] = None,
                 explanation: Optional[str] = None) -> dict:
    """Create Score structure."""
    return {
        "value": value,
        "answer": answer,
        "explanation": explanation,
        "metadata": None,
        "history": []
    }


def create_chat_message(role: str, content: str, source: str = "generate") -> dict:
    """Create ChatMessage structure."""
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "source": source,
        "metadata": None
    }


def create_model_output(completion: str, model: str) -> dict:
    """Create ModelOutput structure."""
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": completion,
                    "source": "generate",
                    "metadata": None,
                    "tool_calls": None
                },
                "stop_reason": "stop",
                "logprobs": None
            }
        ],
        "completion": completion,
        "usage": create_model_usage(),
        "time": 0.5,
        "metadata": None,
        "error": None
    }


def create_eval_sample(
    sample_id: str | int,
    epoch: int,
    input_text: str,
    target: str,
    output_text: str,
    score_value: float,
    model: str,
    metadata: Optional[dict] = None,
    early_stopped: bool = False,
    stop_reason: Optional[str] = None
) -> dict:
    """Create EvalSample structure matching inspect_ai format."""
    messages = [
        create_chat_message("user", input_text, "input"),
        create_chat_message("assistant", output_text, "generate")
    ]

    sample = {
        "id": sample_id,
        "epoch": epoch,
        "input": input_text,
        "choices": None,
        "target": target,
        "sandbox": None,
        "files": None,
        "setup": None,
        "messages": messages,
        "output": create_model_output(output_text, model),
        "scores": {
            "accuracy": create_score(score_value)
        },
        "metadata": metadata or {},
        "store": {},
        "events": [],
        "model_usage": {model: create_model_usage()},
        "total_time": 1.0,
        "working_time": 0.8,
        "uuid": str(uuid.uuid4()),
        "error": None,
        "error_retries": None,
        "attachments": {},
        "limit": None
    }

    # Add early stopping information to sample metadata
    if early_stopped:
        sample["metadata"]["early_stopped"] = True
        sample["metadata"]["stop_reason"] = stop_reason

    return sample


def create_eval_spec(
    task_name: str,
    model: str,
    num_samples: int,
    epochs: int,
    created: str,
    metadata: Optional[dict] = None
) -> dict:
    """Create EvalSpec structure."""
    eval_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    return {
        "eval_set_id": None,
        "eval_id": eval_id,
        "run_id": run_id,
        "created": created,
        "task": task_name,
        "task_id": task_id,
        "task_version": 1,
        "task_file": f"tasks/{task_name}.py",
        "task_display_name": task_name,
        "task_registry_name": task_name,
        "task_attribs": {},
        "task_args": {},
        "task_args_passed": {},
        "solver": "generate",
        "solver_args": None,
        "tags": ["early_stopping_demo"],
        "dataset": {
            "name": f"{task_name}_dataset",
            "location": f"datasets/{task_name}.jsonl",
            "samples": num_samples,
            "sample_ids": None,
            "shuffled": False
        },
        "sandbox": None,
        "model": model,
        "model_generate_config": {
            "max_tokens": 1024,
            "temperature": 0.0
        },
        "model_base_url": None,
        "model_args": {},
        "model_roles": None,
        "config": {
            "limit": None,
            "sample_id": None,
            "sample_shuffle": None,
            "epochs": epochs,
            "epochs_reducer": None,
            "approval": None,
            "fail_on_error": None,
            "continue_on_fail": None,
            "retry_on_error": None,
            "message_limit": None,
            "token_limit": None,
            "time_limit": None,
            "working_limit": None,
            "max_samples": None,
            "max_tasks": None,
            "max_subprocesses": None,
            "max_sandboxes": None,
            "sandbox_cleanup": None,
            "log_samples": True,
            "log_realtime": None,
            "log_images": True,
            "log_buffer": None,
            "log_shared": None,
            "score_display": None
        },
        "revision": None,
        "packages": {
            "inspect_ai": "0.3.152",
            "optstop": "0.3.0"
        },
        "metadata": metadata,
        "scorers": [
            {
                "name": "accuracy",
                "options": None,
                "metrics": None,
                "metadata": None
            }
        ],
        "metrics": None
    }


def create_eval_results(
    total_samples: int,
    completed_samples: int,
    mean_score: float,
    early_stopping_metadata: Optional[dict] = None
) -> dict:
    """Create EvalResults structure."""
    results = {
        "total_samples": total_samples,
        "completed_samples": completed_samples,
        "scores": [
            {
                "name": "accuracy",
                "scorer": "accuracy",
                "reducer": None,
                "scored_samples": completed_samples,
                "unscored_samples": total_samples - completed_samples,
                "params": {},
                "metrics": {
                    "mean": {
                        "name": "mean",
                        "value": mean_score,
                        "params": {},
                        "metadata": None
                    }
                },
                "metadata": None
            }
        ],
        "metadata": early_stopping_metadata
    }
    return results


def create_eval_stats(started_at: str, completed_at: str, model: str) -> dict:
    """Create EvalStats structure."""
    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "model_usage": {
            model: create_model_usage(10000, 5000)
        }
    }


def create_eval_plan() -> dict:
    """Create EvalPlan structure."""
    return {
        "name": "plan",
        "steps": [
            {
                "solver": "generate",
                "params": {}
            }
        ],
        "finish": None,
        "config": {
            "max_tokens": 1024,
            "temperature": 0.0
        }
    }


def create_eval_log(
    eval_spec: dict,
    plan: dict,
    results: dict,
    stats: dict,
    samples: list[dict],
    status: str = "success",
    early_stopping_summary: Optional[dict] = None
) -> dict:
    """Create complete EvalLog structure."""
    log = {
        "version": 2,
        "status": status,
        "eval": eval_spec,
        "plan": plan,
        "results": results,
        "stats": stats,
        "error": None,
        "samples": samples,
        "reductions": None
    }

    # Add early stopping summary to results metadata
    # This is where inspect_ai would store EarlyStopping diagnostics
    if early_stopping_summary:
        if log["results"]["metadata"] is None:
            log["results"]["metadata"] = {}
        log["results"]["metadata"]["early_stopping"] = early_stopping_summary

    return log


# =============================================================================
# Mock Data Generation
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
        # Use stored sample base or create new one
        key = f"ordinal_{sample_id}"
        if key not in self._sample_base_rates:
            rate = base_category + self.rng.uniform(-sample_variance, sample_variance)
            self._sample_base_rates[key] = np.clip(rate, 0, max_score)

        sample_base = self._sample_base_rates[key]
        epoch_noise = self.rng.uniform(-epoch_variance, epoch_variance)
        return int(np.clip(round(sample_base + epoch_noise), 0, max_score))


# =============================================================================
# Run Evaluation with Early Stopping
# =============================================================================

async def run_mock_evaluation(
    task_name: str,
    model: str,
    num_samples: int,
    epochs: int,
    score_type: str,  # 'binary', 'continuous', 'ordinal'
    optstop_params: dict,
    grouping_columns: list[str],
    reanalysis_interval: int = 5,
    seed: int = 42,
    ordinal_max_score: int = 5
) -> tuple[dict, dict]:
    """
    Run a mock evaluation with early stopping and return EvalLog structure.

    Returns:
        Tuple of (eval_log_dict, early_stopping_diagnostics)
    """
    score_generator = MockScoreGenerator(seed=seed)

    # Initialize manager
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=grouping_columns,
        reanalysis_interval=reanalysis_interval,
        min_samples_per_grouping=5,
        ordinal_tasks=['ordinal'] if score_type == 'ordinal' else None,
        ordinal_max_score=ordinal_max_score,
        ordinal_inference='modal' if score_type == 'ordinal' else 'hybrid',
        gpu_ids=None,
        manager_name=f"mock_{task_name}",
        shadow_mode=False,
        score_agg='mean' if score_type == 'continuous' else None,
        random_seed=seed
    )

    # Create mock samples
    mock_samples = []
    for i in range(num_samples):
        sample = Mock()
        sample.id = f"sample_{i:04d}"
        sample.metadata = {"task": task_name}
        mock_samples.append(sample)

    # Create mock EvalSpec
    mock_eval_spec = Mock()
    mock_eval_spec.model = model
    mock_eval_spec.task = task_name
    mock_eval_spec.eval_id = str(uuid.uuid4())
    mock_eval_spec.metadata = {"score_type": score_type}
    mock_eval_spec.tags = ["early_stopping_demo"]

    # Start task
    created_time = datetime.now(timezone.utc).isoformat()
    await manager.start_task(mock_eval_spec, mock_samples, epochs)

    # Track samples and early stops
    eval_samples = []
    stopped_samples_info = {}

    # Run evaluation loop
    for epoch in range(1, epochs + 1):
        for sample in mock_samples:
            sample_id = sample.id

            # Check if should schedule
            early_stop = await manager.schedule_sample(sample_id, epoch)

            if early_stop is not None:
                # Sample was stopped - record info but don't create EvalSample
                if sample_id not in stopped_samples_info:
                    stopped_samples_info[sample_id] = {
                        "stopped_at_epoch": epoch,
                        "reason": early_stop.reason
                    }
                continue

            # Generate score based on type
            if score_type == 'binary':
                score = score_generator.generate_binary_score(sample_id)
            elif score_type == 'continuous':
                score = score_generator.generate_continuous_score(sample_id)
            else:  # ordinal
                score = score_generator.generate_ordinal_score(
                    sample_id, max_score=ordinal_max_score
                )

            # Create mock score object
            mock_score_obj = Mock()
            mock_score_obj.value = score
            mock_sample_score = Mock()
            mock_sample_score.score = mock_score_obj
            mock_scores = {'accuracy': mock_sample_score}

            # Complete sample
            await manager.complete_sample(sample_id, epoch, mock_scores)

            # Create EvalSample for log
            eval_sample = create_eval_sample(
                sample_id=sample_id,
                epoch=epoch,
                input_text=f"Question {sample_id} epoch {epoch}",
                target="correct_answer",
                output_text=f"Model response for {sample_id}",
                score_value=float(score),
                model=model,
                metadata={"task": task_name},
                early_stopped=False
            )
            eval_samples.append(eval_sample)

    # Get early stopping diagnostics
    diagnostics = await manager.complete_task()
    completed_time = datetime.now(timezone.utc).isoformat()

    # Calculate statistics
    total_planned = num_samples * epochs
    total_ran = len(eval_samples)
    mean_score = np.mean([s["scores"]["accuracy"]["value"] for s in eval_samples]) if eval_samples else 0.0

    # Create EvalLog structure
    eval_spec = create_eval_spec(
        task_name=task_name,
        model=model,
        num_samples=num_samples,
        epochs=epochs,
        created=created_time,
        metadata={"score_type": score_type, "early_stopping_enabled": True}
    )

    plan = create_eval_plan()

    results = create_eval_results(
        total_samples=total_planned,
        completed_samples=total_ran,
        mean_score=float(mean_score),
        early_stopping_metadata=diagnostics
    )

    stats = create_eval_stats(created_time, completed_time, model)

    eval_log = create_eval_log(
        eval_spec=eval_spec,
        plan=plan,
        results=results,
        stats=stats,
        samples=eval_samples,
        status="success",
        early_stopping_summary=diagnostics
    )

    return eval_log, diagnostics


# =============================================================================
# Main Script
# =============================================================================

async def generate_all_mock_logs(output_dir: Path, seed: int = 42):
    """Generate all mock eval log files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common parameters
    model = "anthropic/claude-3-sonnet"
    num_samples = 50
    epochs = 20

    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.95,
        'conservatism': 3,
        'tune': 300,
        'draws': 300,
        'chains': 2,
        'cores': 1,
    }

    print("="*80)
    print("Generating Mock inspect_ai Eval Logs with Early Stopping")
    print("="*80)

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
        seed=seed
    )

    binary_path = output_dir / "eval_binary_stopping.json"
    with open(binary_path, 'w') as f:
        json.dump(make_json_serializable(binary_log), f, indent=2)
    print(f"  Saved: {binary_path}")
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
        seed=seed + 1
    )

    continuous_path = output_dir / "eval_continuous_stopping.json"
    with open(continuous_path, 'w') as f:
        json.dump(make_json_serializable(continuous_log), f, indent=2)
    print(f"  Saved: {continuous_path}")
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
        ordinal_max_score=5
    )

    ordinal_path = output_dir / "eval_ordinal_stopping.json"
    with open(ordinal_path, 'w') as f:
        json.dump(make_json_serializable(ordinal_log), f, indent=2)
    print(f"  Saved: {ordinal_path}")
    print(f"  Samples: {ordinal_diag['total_ran']}/{ordinal_diag['total_planned_trials']}")
    print(f"  Efficiency: {ordinal_diag['efficiency_percent']}%")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nGenerated {3} eval log files in: {output_dir}")
    print("\nFiles:")
    print(f"  - eval_binary_stopping.json     ({binary_diag['efficiency_percent']}% efficiency)")
    print(f"  - eval_continuous_stopping.json ({continuous_diag['efficiency_percent']}% efficiency)")
    print(f"  - eval_ordinal_stopping.json    ({ordinal_diag['efficiency_percent']}% efficiency)")

    print("\nEarly Stopping Data Location in Each Log:")
    print("  log['results']['metadata']['early_stopping'] - Full diagnostics from complete_task()")

    print("\nKey Fields in early_stopping:")
    print("  - manager: Name of the early stopping manager")
    print("  - random_seed: MCMC seed for reproducibility")
    print("  - total_planned_trials: Samples × Epochs")
    print("  - total_ran: Actual trials executed")
    print("  - total_skipped: Trials skipped due to early stopping")
    print("  - efficiency_percent: % of trials saved")
    print("  - stopped_samples: List of samples that stopped early")
    print("  - stopped_groupings: Groupings that stopped completely")
    print("  - stabilization_histories: Per-grouping convergence data")

    print("\n" + "="*80)

    return {
        'binary': (binary_log, binary_diag),
        'continuous': (continuous_log, continuous_diag),
        'ordinal': (ordinal_log, ordinal_diag)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate mock inspect_ai eval logs with early stopping data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./test_outputs/mock_eval_logs',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    asyncio.run(generate_all_mock_logs(output_dir, args.seed))


if __name__ == "__main__":
    main()
