"""N identical abc_sim worlds stepped together in MJWarp."""

from __future__ import annotations

import hashlib
from typing import Any

import mujoco
import numpy as np

from abc_sim.env import _GRIPPER_CTRL_MAX, MuJoCoYAMEnv, project_policy_state_batch
from abc_sim.randomization.core import RandomizationSamplingError
from abc_sim.rendering.replay.renderer import (
    PER_EPISODE_MODEL_FIELDS,
    RendererWrapper,
    WarpReplayRuntime,
)
from abc_sim.task_eval import TaskEvalResult, make_task_evaluator
from abc_sim.task_eval.base import evaluate_task_batch

# Compiled geometry every world in a batch must agree on.
SHARED_MODEL_FIELDS = (
    "geom_size", "geom_pos", "geom_quat", "geom_dataid", "geom_type",
    "body_mass", "body_inertia", "mesh_vert", "jnt_qposadr",
)


def model_signature(model: mujoco.MjModel) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for name in SHARED_MODEL_FIELDS:
        digest.update(np.ascontiguousarray(getattr(model, name)).tobytes())
    return digest.digest()


class PerWorldEvaluators:
    """One evaluator per world, for evaluators configured from each episode's randomization."""

    def __init__(self, model: mujoco.MjModel, spec: Any, num_worlds: int):
        self.evaluators = [make_task_evaluator(model, spec) for _ in range(num_worlds)]

    def reset(self, model: mujoco.MjModel, randomizations: list[Any]) -> None:
        for evaluator, state in zip(self.evaluators, randomizations):
            evaluator.reset(nworld=1)
            if state is not None:
                evaluator.configure_from_randomization(model, state)

    def per_world_keys(self, qpos: np.ndarray) -> set[str]:
        metrics = self.evaluators[0].evaluate_qpos_batch(qpos[:1]).metrics
        self.evaluators[0].reset(nworld=1)
        return {key for key, value in metrics.items() if isinstance(value, (list, np.ndarray)) and len(value) == 1}

    def evaluate_qpos_batch(self, qpos: np.ndarray) -> TaskEvalResult:
        results = [
            evaluator.evaluate_qpos_batch(qpos[index : index + 1])
            for index, evaluator in enumerate(self.evaluators)
        ]
        metrics = {}
        for key, value in results[0].metrics.items():
            values = [result.metrics[key] for result in results]
            if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 1:
                metrics[key] = np.concatenate(values)
            elif isinstance(value, list) and len(value) == 1:
                metrics[key] = [item[0] for item in values]
            else:
                metrics[key] = value
        return TaskEvalResult(
            reward=np.concatenate([result.reward for result in results]),
            success=np.concatenate([result.success for result in results]),
            metrics=metrics,
        )


class BatchedWarpYAMEnv:
    """Parallel MJWarp physics and rendering over one abc_sim task.

    Each world is reset through the CPU base env (its randomizer, seeds, and
    evaluator), then its state is loaded into one MJWarp ``Data`` holding
    ``num_worlds`` worlds. All worlds share the compiled model: object poses,
    fixed-body poses, and material colors vary per world, but model-level
    randomization (object scales, mesh variants, object counts) has to be pinned
    through the reset request so every world compiles to the same model.
    Observations carry the camera images as CUDA tensors.
    """

    def __init__(
        self,
        base_env: MuJoCoYAMEnv,
        *,
        num_worlds: int,
        camera_height: int,
        camera_width: int,
        gpu_id: int | None = None,
    ):
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
        if base_env._task_runtime is not None:
            raise NotImplementedError(
                f"task {base_env._task!r} has a per-step task runtime, which is not batched"
            )
        self.base_env = base_env
        self.num_worlds = num_worlds
        self.camera_names = list(base_env.camera_names)
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_decimation = base_env._control_decimation
        self.prompts = [base_env.prompt] * num_worlds
        self.seeds: list[int] = [0] * num_worlds
        self.randomization: list[Any] = [None] * num_worlds
        self.per_world_metric_keys: set[str] = set()
        self.qpos = np.zeros((num_worlds, base_env.model.nq), dtype=np.float32)
        self._gpu_id = gpu_id
        self._signature: bytes | None = None
        self._runtime: WarpReplayRuntime | None = None
        self._renderer: RendererWrapper | None = None
        self._evaluator = None

    @property
    def model(self) -> mujoco.MjModel:
        return self.base_env.model

    def reset(self, seeds: list[int], options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset world i from seeds[i] through the base env and load the batch."""
        if len(seeds) != self.num_worlds:
            raise ValueError(f"expected {self.num_worlds} seeds, got {len(seeds)}")
        snapshots = []
        for index, seed in enumerate(seeds):
            try:
                self.base_env.reset(seed=seed, options=options, randomize=True)
            except RandomizationSamplingError:
                seed += 100_000
                print(f"seed {seed - 100_000} unplaceable, resampled -> {seed}", flush=True)
                self.base_env.reset(seed=seed, options=options, randomize=True)
            self.seeds[index] = seed
            self.randomization[index] = self.base_env._last_randomization
            self.prompts[index] = self.base_env.prompt
            snapshots.append(self._snapshot())
        self._bind_model()
        if any(snapshot["signature"] != self._signature for snapshot in snapshots):
            raise ValueError(
                "worlds in a batch must compile to the same model: pin the task's "
                "model-level randomization (object counts, variants, scales) with "
                "--randomization, e.g. "
                '\'{"bottle_count": 6, "randomize_variants": false, "randomize_scales": false}\''
            )
        stacked = {
            key: np.stack([snapshot[key] for snapshot in snapshots])
            for key in snapshots[0]
            if key != "signature"
        }
        self._runtime.reset_from_mujoco()
        self._runtime.sync_model_from_mujoco(stacked)
        self._runtime.load_state_batch(
            qpos=stacked["qpos"],
            qvel=stacked["qvel"],
            ctrl=stacked["ctrl"],
            act=stacked["act"] if self.model.na else None,
            mocap_pos=stacked["mocap_pos"] if self.model.nmocap else None,
            mocap_quat=stacked["mocap_quat"] if self.model.nmocap else None,
            time=stacked["time"],
        )
        self._runtime.forward()
        self.qpos = stacked["qpos"]
        if isinstance(self._evaluator, PerWorldEvaluators):
            self._evaluator.reset(self.model, self.randomization)
            self.per_world_metric_keys = self._evaluator.per_world_keys(self.qpos)
        elif self._evaluator is not None:
            self.per_world_metric_keys = self._probe_metric_keys()
            self._evaluator.reset(nworld=self.num_worlds)
            for method, key in (
                ("set_active_trash_joints", "trash_joints"),
                ("set_active_object_joints", "active_object_joints"),
            ):
                if hasattr(self._evaluator, method):
                    getattr(self._evaluator, method)([state.metadata[key] for state in self.randomization])
        return self.get_obs()

    def _snapshot(self) -> dict[str, Any]:
        model, data = self.base_env.model, self.base_env.data
        qpos = np.asarray(data.qpos, dtype=np.float32)
        # Randomizers park inactive objects a metre below the floor, in deep
        # penetration with the infinite floor and table planes; the float32
        # solver does not survive that, so rest them on the table plane instead.
        for joint in range(model.njnt):
            adr = model.jnt_qposadr[joint] + 2
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE and qpos[adr] < -0.5:
                qpos[adr] = 0.76
        snapshot = {
            "signature": model_signature(model),
            "qpos": qpos,
            "qvel": np.asarray(data.qvel, dtype=np.float32),
            "ctrl": np.asarray(data.ctrl, dtype=np.float32),
            "act": np.asarray(data.act, dtype=np.float32),
            "mocap_pos": np.asarray(data.mocap_pos, dtype=np.float32),
            "mocap_quat": np.asarray(data.mocap_quat, dtype=np.float32),
            "time": np.float32(data.time),
        }
        for name in PER_EPISODE_MODEL_FIELDS:
            snapshot[name] = np.asarray(getattr(model, name), dtype=np.float32)
        return snapshot

    def _bind_model(self) -> None:
        """Rebuild the MJWarp runtime when the base env compiled a new model."""
        signature = model_signature(self.model)
        if signature == self._signature:
            return
        self._release()
        self._signature = signature
        self._runtime = WarpReplayRuntime(
            self.model,
            self.base_env.data,
            nworld=self.num_worlds,
            gpu_id=self._gpu_id,
            nconmax=4096,
            njmax=32768,
        )
        self._renderer = RendererWrapper(
            backend="mjwarp",
            runtime=self._runtime,
            cam_res=(self.camera_width, self.camera_height),
            gpu_id=self._gpu_id,
        )
        self._camera_index = {self.model.cam(i).name: i for i in range(self.model.ncam)}
        self._evaluator = make_task_evaluator(self.model, self.base_env._task_spec)
        if hasattr(self._evaluator, "configure_from_randomization"):
            self._evaluator = PerWorldEvaluators(self.model, self.base_env._task_spec, self.num_worlds)

    def _probe_metric_keys(self) -> set[str]:
        """Evaluator metric keys with a leading world axis.

        Scene constants (object names, bin geometry) come back beside the
        per-world arrays; scoring two other batch widths tells them apart.
        """
        keys = None
        for width in (self.num_worlds + 1, self.num_worlds + 2):
            result = evaluate_task_batch(
                self._evaluator, np.repeat(self.qpos[:1], width, axis=0), np.zeros(width)
            )
            found = {
                key
                for key, value in result.metrics.items()
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) == width
            }
            keys = found if keys is None else keys & found
        return keys

    def step(self, actions: np.ndarray) -> TaskEvalResult | None:
        """Apply one (num_worlds, 14) action to every world and score the result."""
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_worlds, self.base_env.single_timestep_action_dim)
        if actions.shape != expected:
            raise ValueError(f"expected actions of shape {expected}, got {actions.shape}")
        scaled = actions.copy()
        scaled[:, self.base_env._gripper_indices] *= _GRIPPER_CTRL_MAX
        ctrl = np.zeros((self.num_worlds, self.model.nu), dtype=np.float32)
        ctrl[:, self.base_env._ctrl_indices] = scaled
        self._runtime.set_ctrl_batch(ctrl)
        self._runtime.step(nstep=self.control_decimation)
        data = self._runtime.d_warp
        self.qpos = data.qpos.numpy()
        if int(data.nacon.numpy()[0]) >= data.naconmax:
            raise RuntimeError("MJWarp contact buffer overflowed; raise nconmax in BatchedWarpYAMEnv")
        return self.evaluate()

    def evaluate(self) -> TaskEvalResult | None:
        if self._evaluator is None:
            return None
        return evaluate_task_batch(
            self._evaluator, self.qpos, self._runtime.d_warp.time.numpy()
        )

    def state(self) -> np.ndarray:
        return project_policy_state_batch(
            self.qpos, self.base_env._qpos_indices, self.base_env._gripper_indices
        )

    def render_cameras(self) -> dict[str, Any]:
        """Per-camera (num_worlds, 3, H, W) uint8 CUDA tensors."""
        images = self._renderer.render(actual_batch=self.num_worlds)
        return {
            name: images[:, self._camera_index[name]].permute(0, 3, 1, 2).contiguous()
            for name in self.camera_names
            if name in self._camera_index
        }

    def get_obs(self) -> dict[str, Any]:
        return {
            "state": self.state(),
            "images": self.render_cameras(),
            "prompt": list(self.prompts),
        }

    def _release(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def close(self) -> None:
        self._release()
        self.base_env.close()
