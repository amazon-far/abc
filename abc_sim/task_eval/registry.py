"""Task evaluator registry."""

from __future__ import annotations

from typing import TypeAlias

import mujoco

from abc_sim.task_eval.base import TaskEvaluator
from abc_sim.task_eval.ball_tray import BallTrayBalancingEvaluator
from abc_sim.task_eval.bottles import BottlesInBinEvaluator, PutBottlesInBinEvaluator
from abc_sim.task_eval.chess import ChessBoardSetupEvaluator
from abc_sim.task_eval.conveyor import ConveyorPickObjectsInBinEvaluator
from abc_sim.task_eval.count_box import CountIntoOpaqueBoxEvaluator
from abc_sim.task_eval.dishrack import DishrackEvaluator
from abc_sim.task_eval.grab_clutter import GrabClutterTargetInBoxEvaluator
from abc_sim.task_eval.inhand_transfer import InhandTransferOtherSideEvaluator
from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_eval.mug_tree import MugOnRackEvaluator
from abc_sim.task_eval.multi_drawer import MultiDrawerSearchTargetInBinEvaluator
from abc_sim.task_eval.mugs import MugFlipUprightEvaluator
from abc_sim.task_eval.nuts_bolts import NutsBoltsSortingEvaluator
from abc_sim.task_eval.pour import PourBeadsInContainerEvaluator
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_eval.spell import SpellWordEvaluator
from abc_sim.task_eval.sweep import SweepAwayEvaluator
from abc_sim.task_eval.tower import StackTallestTowerEvaluator
from abc_sim.task_specs import SimTaskSpec, maybe_get_task_spec

TaskEvaluatorFactory: TypeAlias = type[TaskEvaluator]


_TASK_EVALUATORS: dict[str, TaskEvaluatorFactory] = {
    "bottles_in_bin": BottlesInBinEvaluator,
    "put_bottles_in_bin": PutBottlesInBinEvaluator,
    "chess_board_setup": ChessBoardSetupEvaluator,
    "conveyor_pick_objects_in_bin": ConveyorPickObjectsInBinEvaluator,
    "count_into_opaque_box": CountIntoOpaqueBoxEvaluator,
    "dishrack_plates_in_rack": DishrackEvaluator,
    "mug_flip_upright": MugFlipUprightEvaluator,
    "sweep_away": SweepAwayEvaluator,
    "nuts_bolts_sorting": NutsBoltsSortingEvaluator,
    "lego_blocks_sorting": LegoBlocksSortingEvaluator,
    "grab_clutter_target_in_box": GrabClutterTargetInBoxEvaluator,
    "inhand_transfer_other_side": InhandTransferOtherSideEvaluator,
    "mug_on_rack": MugOnRackEvaluator,
    "multi_drawer_target_in_bin": MultiDrawerSearchTargetInBinEvaluator,
    "ball_tray_balancing": BallTrayBalancingEvaluator,
    "pour_beads_in_container": PourBeadsInContainerEvaluator,
    "put_relative": PutRelativeEvaluator,
    "spell_word": SpellWordEvaluator,
    "stack_tallest_tower": StackTallestTowerEvaluator,
}


def make_task_evaluator(
    model: mujoco.MjModel,
    task: str | SimTaskSpec | None,
) -> TaskEvaluator | None:
    """Instantiate a task evaluator for a task name/spec if one is configured."""

    spec = task if isinstance(task, SimTaskSpec) else maybe_get_task_spec(task)
    if spec is None or spec.evaluator_name is None:
        return None

    try:
        evaluator_cls = _TASK_EVALUATORS[spec.evaluator_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task evaluator {spec.evaluator_name!r} for task {spec.name!r}"
        ) from exc

    return evaluator_cls(model=model, spec=spec, **spec.evaluator_kwargs())
