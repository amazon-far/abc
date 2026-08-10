"""CLI plumbing tests for the opt-in geometry export."""

from __future__ import annotations

from pathlib import Path

import export_hf_task
from export_mcap import ExportMcapConfig, _unpack_export_job


def test_geometry_export_is_off_by_default(tmp_path: Path) -> None:
    config = ExportMcapConfig(root=tmp_path, out_dir=tmp_path)

    assert config.derive_ee_poses is False


def test_historical_three_item_export_jobs_remain_compatible() -> None:
    historical = ("episode.mcap", "task", "out")
    current = (*historical, True)

    assert _unpack_export_job(historical) == (*historical, False)
    assert _unpack_export_job(current) == current


def test_hf_wrapper_only_forwards_explicit_geometry_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    def capture(command, *, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(export_hf_task.subprocess, "run", capture)

    export_hf_task.convert(
        export_hf_task.Config(task="example", cache=tmp_path),
        "train",
        tmp_path / "staged",
    )
    export_hf_task.convert(
        export_hf_task.Config(
            task="example",
            cache=tmp_path,
            derive_ee_poses=True,
        ),
        "train",
        tmp_path / "staged",
    )

    assert "--derive-ee-poses" not in commands[0]
    assert commands[1][-1] == "--derive-ee-poses"
