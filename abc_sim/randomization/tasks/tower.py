"""Seeded, separated pickup poses for the fixed four-piece tower scene."""

from abc_sim.randomization.core import PerturbRange, ScalePerturbRange, SceneRandomizer


class StackTallestTowerRandomizer(SceneRandomizer):
    """Scatter the cuboid, two cubes, and roof without changing their geometry."""

    min_clearance_m = 0.11
    max_tries = 400
    perturbations = [
        PerturbRange("sms_base_joint", delta_x=(-0.12, 0.12), delta_y=(-0.26, 0.26)),
        PerturbRange("sms_cube_1_joint", delta_x=(-0.12, 0.12), delta_y=(-0.26, 0.26)),
        PerturbRange("sms_cube_2_joint", delta_x=(-0.12, 0.12), delta_y=(-0.26, 0.26)),
        PerturbRange("sms_roof_joint", delta_x=(-0.12, 0.12), delta_y=(-0.26, 0.26)),
    ]

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        # The base sampler infers scale jitter from pose perturbations when its
        # size list is empty. This task keeps all four block dimensions fixed.
        return []
