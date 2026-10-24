"""Task randomizer registry."""

from __future__ import annotations

from .core import SceneRandomizer
from .tasks.ball_sorting import BallSortingRandomizer
from .tasks.ball_tray import BallTrayBalancingRandomizer
from .tasks.blocks import BlocksRandomizer
from .tasks.bottles import BottlesRandomizer, WaterBottleRandomizer
from .tasks.chess import ChessRandomizer
from .tasks.conveyor_pick import ConveyorPickObjectRandomizer
from .tasks.count_box import CountIntoOpaqueBoxRandomizer
from .tasks.dishrack import DishRackRandomizer
from .tasks.grab_clutter import GrabClutterRandomizer
from .tasks.drawer import DrawerRandomizer
from .tasks.mugs import MugFlipRandomizer, MugTreeRandomizer
from .tasks.marker import MarkerRandomizer
from .tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from .tasks.pour import PourRandomizer
from .tasks.put_relative import PutRelativeRandomizer
from .tasks.sorting import LegoBlocksSortingRandomizer, NutsBoltsSortingRandomizer
from .tasks.sweep import SweepRandomizer
from .tasks.tower import StackTallestTowerRandomizer

_WATER_BOTTLE_RANDOMIZER = WaterBottleRandomizer()

TASK_RANDOMIZERS: dict[str, SceneRandomizer] = {
    "bottles":      BottlesRandomizer(),
    "put_bottles":  _WATER_BOTTLE_RANDOMIZER,
    "water_bottles": _WATER_BOTTLE_RANDOMIZER,
    "marker":       MarkerRandomizer(),
    "pour":         PourRandomizer(),
    "drawer":       DrawerRandomizer(),
    "dishrack":     DishRackRandomizer(),
    "blocks":       BlocksRandomizer(),
    "mug_tree":     MugTreeRandomizer(),
    "mug_flip":     MugFlipRandomizer(),
    "ball_sorting": BallSortingRandomizer(),
    "chess":        ChessRandomizer(),
    "sweep":        SweepRandomizer(),
    "count_into_opaque_box": CountIntoOpaqueBoxRandomizer(),
    "put_relative": PutRelativeRandomizer(),
    "conveyor_pick": ConveyorPickObjectRandomizer(),
    "ball_tray_balancing": BallTrayBalancingRandomizer(),
    "grab_clutter":  GrabClutterRandomizer(),
    "multi_drawer_search": MultiDrawerSearchRandomizer(),
    "nuts_bolts_sorting": NutsBoltsSortingRandomizer(),
    "lego_blocks_sorting": LegoBlocksSortingRandomizer(),
    "stack_tallest_tower": StackTallestTowerRandomizer(),
    # "empty": no free-jointed objects to randomize
}


__all__ = ['TASK_RANDOMIZERS']
