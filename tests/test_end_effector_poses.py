"""Regression tests for the opt-in end-effector pose sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from abc_minimal.end_effector_poses import (
    POSE_SPECS,
    SCHEMA,
    SIDECAR_FILENAME,
    YAMForwardKinematics,
    derive_end_effector_poses,
    sidecar_metadata,
    write_end_effector_sidecar,
    yam_model_path,
    yam_model_sha256,
)

# Literal telemetry and transforms from the first arm-state messages in this
# ungated public ABC mirror (sequence 0, 2025-04-03T17:28:55.850143744Z):
# https://huggingface.co/datasets/Voxel51/ABC-130k/blob/main/data/val/
# fold_and_stack_the_skirts/episode_aa015e3a-8cb9-4e03-93a9-900fac9ae6b8/
# episode.fo.mcap
# Mirror revision: 9659e8ce4b39580f48369cc31bc2e47a217c40e7.
LEFT_JOINTS = np.array(
    [
        -0.48161287861448265,
        0.7116426337071786,
        0.7040131227588304,
        -0.7028686961165764,
        -0.012016479743651942,
        -0.13103685053788006,
    ]
)
LEFT_ARM_LOCAL_TRANSFORM = np.array(
    [
        [
            -0.632531141717365,
            0.3756759502397681,
            0.6773270518510042,
            0.2229318922544835,
        ],
        [
            0.18317820509999394,
            0.9222451495056406,
            -0.3404550328452526,
            -0.11421321941931482,
        ],
        [
            -0.7525623561765962,
            -0.09127685700529321,
            -0.6521644236242606,
            0.1754101887061787,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
RIGHT_JOINTS = np.array(
    [
        0.43926909285114846,
        1.0530632486457616,
        0.7162203402761875,
        -0.7997634851606019,
        -0.03337911039902686,
        0.0997558556496525,
    ]
)
# The released right-arm record is in a shared bimanual frame.  It differs
# from the arm-local model output only by an observed -0.61 m Y translation.
# This fixture removes that translation; production code never applies or
# records the disputed shared-frame transform.
RIGHT_ARM_LOCAL_TRANSFORM = np.array(
    [
        [
            -0.8605964623532097,
            -0.35379918654529247,
            0.3663329968553596,
            0.21752706566379915,
        ],
        [
            -0.2944080765331712,
            0.9325444245486542,
            0.20900904457763797,
            0.1084818535402495,
        ],
        [
            -0.41556902369784254,
            0.0720210513885209,
            -0.906705770743582,
            0.08778088368163128,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, pose[3:])
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = pose[:3]
    return matrix


def test_official_model_matches_public_left_and_right_arm_local_fixtures() -> None:
    fk = YAMForwardKinematics()

    np.testing.assert_allclose(
        pose_matrix(fk.grasp_pose(LEFT_JOINTS)),
        LEFT_ARM_LOCAL_TRANSFORM,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pose_matrix(fk.grasp_pose(RIGHT_JOINTS)),
        RIGHT_ARM_LOCAL_TRANSFORM,
        rtol=0.0,
        atol=1e-12,
    )


def test_missing_topics_are_nan_and_never_marked_valid() -> None:
    ticks = np.array([90, 110, 120], dtype=np.int64)
    scalars = {
        "/left-arm-state": [(100, LEFT_JOINTS)],
        # Present but malformed is invalid just like an absent source sample.
        "/right-arm-action": [(100, np.zeros(5))],
    }

    result = derive_end_effector_poses(scalars, ticks)

    assert result["timestamp_ns"].dtype == np.int64
    assert result["valid_mask"].dtype == np.uint8
    np.testing.assert_array_equal(
        result["valid_mask"],
        np.array([0, 1, 1], dtype=np.uint8),
    )
    assert np.isnan(result["left_arm_state_pose"][0]).all()
    assert np.isfinite(result["left_arm_state_pose"][1:]).all()
    for key in (
        "right_arm_state_pose",
        "left_arm_action_pose",
        "right_arm_action_pose",
    ):
        assert np.isnan(result[key]).all()


def test_all_four_validity_bits_and_causal_alignment() -> None:
    ticks = np.array([101, 201], dtype=np.int64)
    scalars = {
        topic: [(100, LEFT_JOINTS), (200, RIGHT_JOINTS)] for _, topic, _ in POSE_SPECS
    }

    result = derive_end_effector_poses(scalars, ticks)

    np.testing.assert_array_equal(
        result["valid_mask"], np.array([15, 15], dtype=np.uint8)
    )
    for key, _, _ in POSE_SPECS:
        np.testing.assert_allclose(
            result[key][0], YAMForwardKinematics().grasp_pose(LEFT_JOINTS)
        )
        np.testing.assert_allclose(
            result[key][1], YAMForwardKinematics().grasp_pose(RIGHT_JOINTS)
        )


def test_sidecar_and_episode_metadata_are_self_describing(tmp_path: Path) -> None:
    ticks = np.array([101], dtype=np.int64)
    # An interrupted or repeated export may already have a target file.  The
    # writer must atomically replace it and leave no temporary artifact.
    (tmp_path / SIDECAR_FILENAME).write_bytes(b"stale")
    metadata = write_end_effector_sidecar(
        tmp_path,
        {"/left-arm-state": [(100, LEFT_JOINTS)]},
        ticks,
    )

    with np.load(tmp_path / SIDECAR_FILENAME) as sidecar:
        assert set(sidecar.files) == {
            *(key for key, _, _ in POSE_SPECS),
            "timestamp_ns",
            "valid_mask",
        }
        assert sidecar["left_arm_state_pose"].shape == (1, 7)
    assert {path.name for path in tmp_path.iterdir()} == {SIDECAR_FILENAME}
    assert metadata == sidecar_metadata()
    assert metadata["schema"] == SCHEMA
    assert metadata["frame"] == "arm_local"
    assert metadata["provenance"]["mujoco_version"] == mujoco.__version__
    assert metadata["model"]["sha256"] == yam_model_sha256()
    assert len(metadata["model"]["sha256"]) == 64
    assert yam_model_path().is_file()
    # Ensure the descriptor remains JSON serializable before the exporter
    # inserts it into episode_metadata.json.
    json.dumps(metadata)
