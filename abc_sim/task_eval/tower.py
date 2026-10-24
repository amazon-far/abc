"""Contact-based four-block tower success with a simulation-time dwell window."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec

PIECE_NAMES = ("sms_base", "sms_cube_1", "sms_cube_2", "sms_roof")


class StackTallestTowerEvaluator:
    """Require all four pieces in one supported, released tower for five seconds.

    Support comes from upward-facing MuJoCo contacts, so tipped pieces and the
    triangular prism are scored using their actual collision geometry. No shape
    order is prescribed. Position drift is measured from the start of the dwell
    window; any lost support, grasp, or excess drift clears current success.
    """

    def __init__(self, *, model: mujoco.MjModel, spec: SimTaskSpec) -> None:
        self.model = model
        self.spec = spec
        self.stability_seconds = 5.0
        self.max_drift_m = 0.003
        self.max_rotation_rad = np.deg2rad(5.0)
        self._bodies = np.array([model.body(name).id for name in PIECE_NAMES])
        self._geoms = {
            model.geom(name + "_col").id: i for i, name in enumerate(PIECE_NAMES)
        }
        self._qpos_adr = np.array(
            [model.joint(name + "_joint").qposadr[0] for name in PIECE_NAMES]
        )
        self._table = model.geom("table_plane").id
        fingers = [model.joint(side + "_left_finger") for side in ("left", "right")]
        self._finger_adr = np.array([joint.qposadr[0] for joint in fingers])
        self._open_threshold = np.array([0.6 * joint.range[1] for joint in fingers])
        self._scratch = mujoco.MjData(model)
        self.reset()

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = nworld
        self._last_time = np.full(nworld, np.nan)
        self._hold_since = np.full(nworld, np.nan)
        self._reference_pos = np.zeros((nworld, 4, 3))
        self._reference_quat = np.zeros((nworld, 4, 4))

    def evaluate_model_data(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> TaskEvalResult:
        if model is not self.model:
            raise ValueError("Create a new tower evaluator after replacing the model")
        # Refresh contacts from qpos: mj_step's derived fields can precede its
        # final integration, and callers may have edited poses before scoring.
        return self.evaluate_timed_qpos_batch(data.qpos[None, :], np.array([data.time]))

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        raise ValueError(
            "Tower stability requires simulation time; use evaluate_timed_qpos_batch"
        )

    def evaluate_timed_qpos_batch(
        self, qpos_batch: np.ndarray, simulation_time: np.ndarray
    ) -> TaskEvalResult:
        """Score each world at its actual simulator timestamp, in seconds."""
        qpos = np.asarray(qpos_batch, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos shape (B, {self.model.nq}), got {qpos.shape}"
            )
        times = np.asarray(simulation_time, dtype=np.float64)
        if times.shape != (len(qpos),) or not np.all(np.isfinite(times)):
            raise ValueError(
                "simulation_time must contain one finite timestamp per world"
            )
        if not np.all(np.isfinite(qpos)):
            raise ValueError("Tower qpos must be finite")
        if len(qpos) != self._nworld:
            self.reset(nworld=len(qpos))
        if np.any(times < self._last_time):
            raise ValueError(
                "Simulation time moved backwards; reset the evaluator first"
            )

        success = np.zeros(len(qpos), dtype=bool)
        released = np.zeros(len(qpos), dtype=bool)
        counts = np.zeros(len(qpos), dtype=np.int32)
        stable_seconds = np.zeros(len(qpos))
        for world, (pose, now) in enumerate(zip(qpos, times)):
            self._scratch.qpos[:] = pose
            mujoco.mj_forward(self.model, self._scratch)
            count, free = self._tower_state(self._scratch)
            counts[world], released[world] = count, free
            positions = np.stack([pose[a : a + 3] for a in self._qpos_adr])
            quats = np.stack([pose[a + 3 : a + 7] for a in self._qpos_adr])
            quats /= np.linalg.norm(quats, axis=1, keepdims=True)
            holding = np.isfinite(self._hold_since[world])
            drift = np.linalg.norm(positions - self._reference_pos[world], axis=1).max()
            dots = np.abs(np.sum(quats * self._reference_quat[world], axis=1))
            rotation = 2 * np.arccos(np.clip(dots, 0.0, 1.0)).max()
            goal = count == len(PIECE_NAMES) and free
            if not goal:
                self._hold_since[world] = np.nan
            else:
                if (
                    not holding
                    or drift > self.max_drift_m
                    or rotation > self.max_rotation_rad
                ):
                    self._hold_since[world] = now
                    self._reference_pos[world] = positions
                    self._reference_quat[world] = quats
                stable_seconds[world] = now - self._hold_since[world]
                success[world] = stable_seconds[world] + 1e-10 >= self.stability_seconds
            self._last_time[world] = now

        reward = (
            0.5 * counts / 4
            + 0.25 * released
            + 0.25 * np.minimum(stable_seconds / 5, 1)
        )
        reward = np.where(success, 1.0, np.minimum(reward, 0.99))
        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "num_blocks_in_tower": counts,
                "released": released,
                "goal_satisfied_now": (counts == 4) & released,
                "stable_seconds": stable_seconds,
                "required_stable_seconds": self.stability_seconds,
            },
        )

    def _tower_state(self, data: mujoco.MjData) -> tuple[int, bool]:
        # supports[upper, lower] is directed upward from the table. Horizontal
        # contact between adjacent blocks does not join them into a tower.
        supports = np.zeros((4, 4), dtype=bool)
        on_table = np.zeros(4, dtype=bool)
        released = bool(np.all(data.qpos[self._finger_adr] >= self._open_threshold))
        for contact in data.contact:
            if contact.dist > 0.001:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            a, b = self._geoms.get(first), self._geoms.get(second)
            if a is None and b is None:
                continue
            if a is not None and b is not None:
                normal_z = float(contact.frame[2])  # geom1 -> geom2
                if (
                    normal_z > 0.5
                    and data.xpos[self._bodies[b], 2] > data.xpos[self._bodies[a], 2]
                ):
                    supports[b, a] = True
                elif (
                    normal_z < -0.5
                    and data.xpos[self._bodies[a], 2] > data.xpos[self._bodies[b], 2]
                ):
                    supports[a, b] = True
            else:
                piece = a if a is not None else b
                other = second if a is not None else first
                if other == self._table:
                    on_table[piece] = True
                else:
                    # A robot, wall, or other external prop is still touching
                    # the tower, even if both grippers have opened.
                    released = False

        # A piece already resting on the table is a separate base; a sloping
        # side contact with its neighbor must not join two tabletop pieces.
        supports[on_table, :] = False
        rooted = on_table.copy()
        for _ in PIECE_NAMES:
            rooted |= np.any(supports & rooted[None, :], axis=1)
        adjacency = supports | supports.T
        best = 0
        for start in np.flatnonzero(on_table):
            component = {int(start)}
            pending = [int(start)]
            while pending:
                node = pending.pop()
                for neighbor in np.flatnonzero(adjacency[node] & rooted):
                    if int(neighbor) not in component:
                        component.add(int(neighbor))
                        pending.append(int(neighbor))
            best = max(best, len(component))
        return best, released
