"""Shared task-resolution helpers for env construction and replay."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from abc_sim.task_specs import SimTaskSpec, maybe_get_task_spec


_MODELS_DIR = Path(__file__).parent / "models"


@dataclass(frozen=True)
class SceneTaskSpec:
    """Low-level scene task definition used by the simulator runtime."""

    name: str
    scene_xml: Path
    physics_defaults: tuple[tuple[str, Any], ...] = ()

    def physics_kwargs(self) -> dict[str, Any]:
        return dict(self.physics_defaults)


@dataclass(frozen=True)
class ResolvedTask:
    """Resolved task metadata spanning prompt specs and scene/runtime data."""

    request_name: str | None
    env_task: str | None
    task_spec: SimTaskSpec | None
    scene_task: SceneTaskSpec | None


_SCENE_TASKS: tuple[SceneTaskSpec, ...] = (
    SceneTaskSpec(name="bottles", scene_xml=_MODELS_DIR / "yam_bottles_scene.xml"),
    SceneTaskSpec(name="put_bottles", scene_xml=_MODELS_DIR / "yam_put_bottles_scene.xml"),
    SceneTaskSpec(name="conveyor_pick", scene_xml=_MODELS_DIR / "yam_conveyor_pick_scene.xml"),
    SceneTaskSpec(name="count_into_opaque_box", scene_xml=_MODELS_DIR / "yam_count_into_opaque_box_scene.xml"),
    SceneTaskSpec(name="put_relative", scene_xml=_MODELS_DIR / "yam_put_relative_scene.xml"),
    SceneTaskSpec(name="grab_clutter", scene_xml=_MODELS_DIR / "yam_grab_clutter_scene.xml"),
    SceneTaskSpec(name="marker", scene_xml=_MODELS_DIR / "yam_marker_scene.xml"),
    SceneTaskSpec(name="ball_sorting", scene_xml=_MODELS_DIR / "yam_ball_sorting_scene.xml"),
    SceneTaskSpec(name="ball_tray_balancing", scene_xml=_MODELS_DIR / "yam_ball_tray_balancing_scene.xml"),
    SceneTaskSpec(name="empty", scene_xml=_MODELS_DIR / "yam_bimanual_empty.xml"),
    SceneTaskSpec(name="dishrack", scene_xml=_MODELS_DIR / "yam_dishrack_base.xml"),
    SceneTaskSpec(name="chess", scene_xml=_MODELS_DIR / "yam_chess_scene.xml"),
    SceneTaskSpec(name="chess_flexible", scene_xml=_MODELS_DIR / "yam_flexible_chess_scene.xml"),
    SceneTaskSpec(name="blocks", scene_xml=_MODELS_DIR / "yam_blocks_scene.xml"),
    SceneTaskSpec(name="mug_tree", scene_xml=_MODELS_DIR / "yam_mug_tree_scene.xml"),
    SceneTaskSpec(name="mug_flip", scene_xml=_MODELS_DIR / "yam_mug_flip_scene.xml"),
    # Keep physics_dt * control_decimation near 1/30 s for policy timing.
    SceneTaskSpec(
        name="pour",
        scene_xml=_MODELS_DIR / "yam_pour_screw_scene.xml",
        physics_defaults=(("physics_dt", 0.0005), ("control_decimation", 68)),
    ),
    SceneTaskSpec(
        name="drawer",
        scene_xml=_MODELS_DIR / "yam_drawer_scene.xml",
        physics_defaults=(("physics_dt", 0.0002), ("control_decimation", 170)),
    ),
    SceneTaskSpec(
        name="multi_drawer_search",
        scene_xml=_MODELS_DIR / "yam_multi_drawer_search_scene.xml",
    ),
    SceneTaskSpec(name="jenga", scene_xml=_MODELS_DIR / "yam_jenga_scene.xml"),
    SceneTaskSpec(
        name="stack_tallest_tower",
        scene_xml=_MODELS_DIR / "yam_stack_tallest_tower_scene.xml",
    ),
    SceneTaskSpec(name="building_blocks", scene_xml=_MODELS_DIR / "yam_building_blocks_scene.xml"),
    SceneTaskSpec(name="sweep", scene_xml=_MODELS_DIR / "yam_sweep_scene.xml"),
    SceneTaskSpec(name="nuts_bolts_sorting", scene_xml=_MODELS_DIR / "yam_nuts_bolts_sorting_scene.xml"),
    SceneTaskSpec(name="lego_blocks_sorting", scene_xml=_MODELS_DIR / "yam_lego_blocks_sorting_scene.xml"),
    SceneTaskSpec(name="inhand_transfer", scene_xml=_MODELS_DIR / "yam_inhand_transfer_base.xml"),
)

_SCENE_TASK_LOOKUP = {spec.name: spec for spec in _SCENE_TASKS}
_SCENE_TASK_ALIASES = {
    "water_bottles": "put_bottles",
}

SCENE_XMLS: dict[str, Path] = {
    spec.name: spec.scene_xml
    for spec in _SCENE_TASKS
}

DEFAULT_SCENE_XML = Path(
    os.environ.get("MUJOCO_SCENE_XML", str(SCENE_XMLS["bottles"]))
)


def list_scene_tasks() -> tuple[SceneTaskSpec, ...]:
    """Return all known low-level scene tasks."""

    return _SCENE_TASKS


def list_scene_task_names() -> tuple[str, ...]:
    """Return all valid low-level scene task names."""

    return tuple(spec.name for spec in _SCENE_TASKS)


def resolve_env_task_name(task: str | SimTaskSpec | None) -> str | None:
    """Resolve a user-facing task or prompt alias into a scene task name."""

    if task is None:
        return None
    if isinstance(task, SimTaskSpec):
        return _SCENE_TASK_ALIASES.get(task.env_task, task.env_task)

    spec = maybe_get_task_spec(task)
    if spec is not None:
        return _SCENE_TASK_ALIASES.get(spec.env_task, spec.env_task)
    return _SCENE_TASK_ALIASES.get(task, task)


def maybe_get_scene_task_spec(task: str | SimTaskSpec | None) -> SceneTaskSpec | None:
    """Resolve the low-level scene task definition, if available."""

    env_task = resolve_env_task_name(task)
    if env_task is None:
        return None
    return _SCENE_TASK_LOOKUP.get(env_task)


def get_scene_task_spec(task: str | SimTaskSpec) -> SceneTaskSpec:
    """Resolve the low-level scene task definition or raise."""

    spec = maybe_get_scene_task_spec(task)
    if spec is None:
        available = ", ".join(list_scene_task_names())
        raise KeyError(f"Unknown scene task {task!r}. Available: {available}")
    return spec


def resolve_task(task: str | SimTaskSpec | None) -> ResolvedTask:
    """Resolve prompt/task aliases into both task-spec and scene-task data."""

    request_name: str | None
    if task is None:
        request_name = None
    elif isinstance(task, SimTaskSpec):
        request_name = task.name
    else:
        request_name = task

    task_spec = task if isinstance(task, SimTaskSpec) else maybe_get_task_spec(task)
    scene_task = maybe_get_scene_task_spec(task_spec or task)
    env_task = scene_task.name if scene_task is not None else resolve_env_task_name(task_spec or task)
    return ResolvedTask(
        request_name=request_name,
        env_task=env_task,
        task_spec=task_spec,
        scene_task=scene_task,
    )


def get_task_scene_xml(task: str | SimTaskSpec | None) -> Path | None:
    """Return the XML path for a task if it maps to a known scene."""

    scene_task = maybe_get_scene_task_spec(task)
    return None if scene_task is None else scene_task.scene_xml


def get_task_physics_defaults(task: str | SimTaskSpec | None) -> dict[str, Any]:
    """Return task-specific physics overrides for a scene task."""

    scene_task = maybe_get_scene_task_spec(task)
    if scene_task is None:
        return {}
    return scene_task.physics_kwargs()


def get_task_randomizer(task: str | SimTaskSpec | None) -> Any:
    """Return the scene randomizer associated with a task, if any."""

    from abc_sim.randomization import TASK_RANDOMIZERS

    env_task = resolve_env_task_name(task)
    if env_task is None:
        return None
    return TASK_RANDOMIZERS.get(env_task)


__all__ = [
    "DEFAULT_SCENE_XML",
    "ResolvedTask",
    "SCENE_XMLS",
    "SceneTaskSpec",
    "get_scene_task_spec",
    "get_task_physics_defaults",
    "get_task_randomizer",
    "get_task_scene_xml",
    "list_scene_task_names",
    "list_scene_tasks",
    "maybe_get_scene_task_spec",
    "resolve_env_task_name",
    "resolve_task",
]
