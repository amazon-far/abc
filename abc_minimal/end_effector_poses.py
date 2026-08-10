"""Derive YAM grasp-site poses from aligned ABC arm joint telemetry.

The exporter deliberately emits these values as a separate, opt-in sidecar.
It does not change the 28-D state/action training tensor, and it does not guess
the transform between the two physical arm bases.  Every pose is expressed in
the local base frame of the arm that produced the joint values.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Final

import mujoco
import numpy as np

SCHEMA: Final = "abc.end_effector_poses.v1"
SIDECAR_FILENAME: Final = "end_effector_poses.npz"
POSE_FORMAT: Final = "xyz_quaternion_wxyz"
POSE_FRAME: Final = "arm_local"
SITE_NAME: Final = "grasp_site"
MODEL_RELATIVE_PATH: Final = Path("assets/put_bottles/assets/i2rt_yam/yam.xml")

# The bit assignments are part of the v1 schema.  Keep their order stable.
POSE_SPECS: Final = (
    ("left_arm_state_pose", "/left-arm-state", 0),
    ("right_arm_state_pose", "/right-arm-state", 1),
    ("left_arm_action_pose", "/left-arm-action", 2),
    ("right_arm_action_pose", "/right-arm-action", 3),
)


def yam_model_path() -> Path:
    """Return the official YAM MJCF bundled with this repository."""

    return Path(__file__).resolve().parents[1] / MODEL_RELATIVE_PATH


@lru_cache(maxsize=1)
def yam_model_sha256() -> str:
    """Return a content digest for the exact MJCF used for derivation."""

    return hashlib.sha256(yam_model_path().read_bytes()).hexdigest()


class YAMForwardKinematics:
    """MuJoCo forward kinematics for the bundled six-joint YAM arm chain."""

    def __init__(self) -> None:
        model_path = yam_model_path()
        if not model_path.is_file():
            raise FileNotFoundError(f"bundled YAM model not found: {model_path}")

        # Loading from the file path, rather than an XML string, lets MuJoCo
        # resolve the model's vendored mesh assets relative to yam.xml.
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"joint{joint_index}",
                )
                for joint_index in range(1, 7)
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            raise RuntimeError("bundled YAM model is missing one or more arm joints")
        self._qpos_addresses = self.model.jnt_qposadr[joint_ids].copy()

        self._site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            SITE_NAME,
        )
        if self._site_id < 0:
            raise RuntimeError(f"bundled YAM model is missing {SITE_NAME}")

    def grasp_pose(self, joints: np.ndarray) -> np.ndarray:
        """Return ``[x, y, z, qw, qx, qy, qz]`` in the arm-local frame."""

        values = np.asarray(joints, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("joints must be six finite values in radians")

        self.data.qpos[self._qpos_addresses] = values
        mujoco.mj_forward(self.model, self.data)

        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self._site_id])
        quaternion /= np.linalg.norm(quaternion)
        # q and -q represent the same rotation.  Canonicalizing the sign keeps
        # serialized results stable across MuJoCo versions.
        if quaternion[0] < 0:
            quaternion = -quaternion
        return np.concatenate((self.data.site_xpos[self._site_id].copy(), quaternion))


@lru_cache(maxsize=1)
def _kinematics() -> YAMForwardKinematics:
    """Cache one compiled model and mutable data object per exporter process."""

    return YAMForwardKinematics()


def _floor_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    # Keep -1 for targets before the first source sample.  The main exporter
    # starts after every active stream, but retaining this distinction here
    # prevents a future caller from silently using a sample from the future.
    return np.searchsorted(source_ts, target_ts, side="right") - 1


def derive_end_effector_poses(
    scalars: dict[str, list[tuple[int, np.ndarray]]],
    timestamp_ns: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive aligned arm-local poses and a source-validity mask.

    Missing topics and malformed/non-finite joint samples remain NaN and leave
    the corresponding validity bit clear.  In particular, this function never
    applies FK to the zero-padding used by the legacy training tensor.
    """

    ticks = np.asarray(timestamp_ns, dtype=np.int64)
    if ticks.ndim != 1:
        raise ValueError(f"timestamp_ns must be one-dimensional, got {ticks.shape}")

    poses = {
        key: np.full((len(ticks), 7), np.nan, dtype=np.float64)
        for key, _, _ in POSE_SPECS
    }
    valid_mask = np.zeros(len(ticks), dtype=np.uint8)

    for key, topic, bit_index in POSE_SPECS:
        messages = scalars.get(topic)
        if not messages:
            continue

        messages = sorted(messages, key=lambda item: item[0])
        source_ts = np.asarray([time_ns for time_ns, _ in messages], dtype=np.int64)
        selected = _floor_indices(source_ts, ticks)
        output = poses[key]
        for row_index, source_index in enumerate(selected):
            if source_index < 0:
                continue
            joints = np.asarray(messages[int(source_index)][1], dtype=np.float64)
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                continue
            output[row_index] = _kinematics().grasp_pose(joints)
            valid_mask[row_index] |= np.uint8(1 << bit_index)

    return {**poses, "timestamp_ns": ticks.copy(), "valid_mask": valid_mask}


def sidecar_metadata() -> dict:
    """Return the self-describing metadata stored in episode_metadata.json."""

    return {
        "schema": SCHEMA,
        "file": SIDECAR_FILENAME,
        "pose_arrays": [key for key, _, _ in POSE_SPECS],
        "pose_format": POSE_FORMAT,
        "frame": POSE_FRAME,
        "site": SITE_NAME,
        "units": {
            "position": "meter",
            "joint_angle": "radian",
            "timestamp": "nanosecond",
        },
        "valid_mask_bits": {
            str(bit_index): {"array": key, "source_topic": topic}
            for key, topic, bit_index in POSE_SPECS
        },
        "provenance": {
            "method": "mujoco.mj_forward",
            "mujoco_version": mujoco.__version__,
            "joint_alignment": "fixed_clock_30hz_causal_floor",
            "source": "raw MCAP arm joint telemetry",
        },
        "model": {
            "path": MODEL_RELATIVE_PATH.as_posix(),
            "sha256": yam_model_sha256(),
        },
    }


def write_end_effector_sidecar(
    out_dir: Path,
    scalars: dict[str, list[tuple[int, np.ndarray]]],
    timestamp_ns: np.ndarray,
) -> dict:
    """Write ``end_effector_poses.npz`` and return its episode metadata."""

    arrays = derive_end_effector_poses(scalars, timestamp_ns)
    out_dir = Path(out_dir)
    target = out_dir / SIDECAR_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=out_dir,
            prefix=f".{SIDECAR_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return sidecar_metadata()


__all__ = [
    "MODEL_RELATIVE_PATH",
    "POSE_FORMAT",
    "POSE_FRAME",
    "POSE_SPECS",
    "SCHEMA",
    "SIDECAR_FILENAME",
    "SITE_NAME",
    "YAMForwardKinematics",
    "derive_end_effector_poses",
    "sidecar_metadata",
    "write_end_effector_sidecar",
    "yam_model_path",
    "yam_model_sha256",
]
