"""Task evaluation interfaces for abc_sim tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from abc_sim.task_specs import SimTaskSpec


def _normalize_info_value(value: Any, *, squeeze: bool) -> Any:
    if isinstance(value, np.ndarray):
        if squeeze and value.ndim > 0 and value.shape[0] == 1:
            return _normalize_info_value(value[0], squeeze=False)
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        if squeeze and len(value) == 1:
            return value[0]
        return value
    if isinstance(value, tuple):
        return [_normalize_info_value(item, squeeze=squeeze) for item in value]
    return value


@dataclass(frozen=True)
class TaskEvalResult:
    """Evaluation output for one or more worlds."""

    reward: np.ndarray
    success: np.ndarray
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        reward = np.asarray(self.reward, dtype=np.float32)
        success = np.asarray(self.success, dtype=bool)
        if reward.ndim != 1:
            raise ValueError(f"Expected reward shape (B,), got {reward.shape}")
        if success.shape != reward.shape:
            raise ValueError(
                f"Reward and success must have matching shapes, got {reward.shape} and {success.shape}"
            )
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "success", success)

    @property
    def num_worlds(self) -> int:
        return int(self.reward.shape[0])

    def scalar_reward(self) -> float:
        if self.num_worlds != 1:
            raise ValueError("scalar_reward() only applies to single-world results")
        return float(self.reward[0])

    def scalar_success(self) -> bool:
        if self.num_worlds != 1:
            raise ValueError("scalar_success() only applies to single-world results")
        return bool(self.success[0])

    def to_info(self, *, squeeze: bool = False) -> dict[str, Any]:
        """Convert metrics into a JSON-friendly dict for env info payloads."""

        info = {
            "reward": self.reward,
            "success": self.success,
            **self.metrics,
        }
        return {
            key: _normalize_info_value(value, squeeze=squeeze)
            for key, value in info.items()
        }


class TaskEvaluator(Protocol):
    """Task evaluator that can score one or more worlds from qpos."""

    spec: SimTaskSpec

    def reset(self, *, nworld: int = 1) -> None:
        """Reset any per-episode evaluator state."""

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        """Evaluate reward/success for a qpos batch of shape ``(B, nq)``."""


def evaluate_task_batch(
    evaluator: TaskEvaluator, qpos: np.ndarray, simulation_time: np.ndarray
) -> TaskEvalResult:
    """Pass simulator timestamps to evaluators with a timed scoring hook.

    Pose-only evaluators retain their existing interface. Stateful dwell tasks
    implement ``evaluate_timed_qpos_batch(qpos, simulation_time)`` so polling
    cannot advance their timers and different physics rates score consistently.
    """
    timed = getattr(evaluator, "evaluate_timed_qpos_batch", None)
    if timed is not None:
        return timed(qpos, simulation_time)
    return evaluator.evaluate_qpos_batch(qpos)
