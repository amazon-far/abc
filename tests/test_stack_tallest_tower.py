"""CPU regression tests; prepare assets with prepare.py --sim-task stack_tallest_tower."""

from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

import abc_sim
from abc_sim.batched_env import BatchedWarpYAMEnv
from abc_sim.task_eval.base import TaskEvalResult, evaluate_task_batch
from abc_sim.task_eval.registry import make_task_evaluator
from abc_sim.task_eval.tower import PIECE_NAMES
from abc_sim.task_registry import get_task_scene_xml


class TowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = abc_sim.make_env(task="stack_tallest_tower", render_cameras=False)
        cls.model = cls.env.model

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(randomize=False)
        self.model = self.env.model
        self.data = self.env.data
        self.evaluator = self.env._task_evaluator
        self.evaluator.reset()
        self.data.time = 0
        for side in ("left", "right"):
            self.data.qpos[self.model.joint(side + "_left_finger").qposadr[0]] = 0.0475
            self.data.qpos[
                self.model.joint(side + "_right_finger").qposadr[0]
            ] = -0.0475
            self.data.ctrl[self.model.actuator(side + "_gripper").id] = 0.0475

    def pose(self, name, position, quat=(1, 0, 0, 0)):
        adr = self.model.joint(name + "_joint").qposadr[0]
        self.data.qpos[adr : adr + 7] = [*position, *quat]

    def stack(self, order=PIECE_NAMES):
        z = 0.75
        heights = {
            "sms_base": 0.025,
            "sms_cube_1": 0.04,
            "sms_cube_2": 0.04,
            "sms_roof": 0.025,
        }
        for name in order:
            height = heights[name]
            # Small penetration represents the contact compression after settling.
            z -= 0.0001
            self.pose(name, [0.6, 0, z + height / 2])
            z += height

    def at(self, time):
        self.data.time = time
        return self.env.evaluate_task()

    def test_catalog_and_asset_resolution(self):
        spec = abc_sim.get_task_spec("stack tower")
        self.assertEqual(spec.name, "stack_tallest_tower")
        self.assertIn(spec, abc_sim.list_task_specs())
        self.assertIn("all the available blocks", self.env.prompt)
        self.assertNotIn("roof", self.env.prompt.lower())
        self.assertEqual(
            get_task_scene_xml(spec).name, "yam_stack_tallest_tower_scene.xml"
        )
        from prepare import sim_task_packages

        self.assertEqual(sim_task_packages([spec.name]), {"i2rt_yam"})
        self.assertEqual(self.env.action_space.shape, (14,))
        for name in ("top", "left", "right"):
            self.assertGreaterEqual(self.model.camera(name).id, 0)

    def test_seeded_reset_and_fixed_geometry(self):
        model = self.env.model
        self.env.reset(seed=119, randomize=True)
        first = self.env.data.qpos.copy()
        first_geom = self.env.model.geom_size.copy()
        self.env.reset(seed=119, randomize=True)
        np.testing.assert_allclose(self.env.data.qpos, first, atol=1e-12, rtol=0)
        self.env.reset(seed=120, randomize=True)
        self.assertFalse(np.allclose(self.env.data.qpos, first))
        self.assertIs(self.env.model, model)
        self.assertEqual(self.env._last_randomization.scale_states, {})
        np.testing.assert_array_equal(self.env.model.geom_size, first_geom)
        self.assertFalse(self.at(10).scalar_success())

    def test_all_four_and_exact_five_seconds(self):
        self.stack()
        result = self.at(0)
        self.assertEqual(result.metrics["num_blocks_in_tower"][0], 4)
        self.assertTrue(result.metrics["released"][0])
        for _ in range(100):
            self.assertFalse(self.at(0).scalar_success())
        self.assertFalse(self.at(4.999).scalar_success())
        self.assertTrue(self.at(5).scalar_success())
        self.assertEqual(self.at(5).scalar_reward(), 1.0)

    def test_partial_and_split_towers_fail(self):
        for count in (2, 3):
            with self.subTest(count=count):
                self.setUp()
                self.stack(PIECE_NAMES[:count])
                self.assertEqual(self.at(0).metrics["num_blocks_in_tower"][0], count)
                self.assertFalse(self.at(20).scalar_success())
        self.setUp()
        self.pose("sms_base", [0.48, 0.2, 0.7624])
        self.pose("sms_roof", [0.48, 0.2, 0.7873])
        self.pose("sms_cube_1", [0.6, 0, 0.7699])
        self.pose("sms_cube_2", [0.6, 0, 0.8098])
        self.assertEqual(self.at(0).metrics["num_blocks_in_tower"][0], 2)
        self.assertFalse(self.at(20).scalar_success())

    def test_every_block_is_required_and_success_clears(self):
        for name in PIECE_NAMES:
            with self.subTest(missing=name):
                self.setUp()
                self.stack()
                self.at(0)
                self.assertTrue(self.at(5).scalar_success())
                self.pose(name, [0.8, 0.3, 0.9])
                self.assertFalse(self.at(5.01).scalar_success())
                self.stack()
                self.assertFalse(self.at(5.02).scalar_success())
                self.assertFalse(self.at(10.019).scalar_success())
                self.assertTrue(self.at(10.02).scalar_success())

    def test_either_closed_gripper_resets_the_window(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                self.setUp()
                self.stack()
                self.at(0)
                self.assertTrue(self.at(5).scalar_success())
                adr = self.model.joint(side + "_left_finger").qposadr[0]
                self.data.qpos[adr] = 0
                self.assertFalse(self.at(5.01).scalar_success())
                self.data.qpos[adr] = 0.0475
                self.assertFalse(self.at(10).scalar_success())
                self.assertTrue(self.at(15).scalar_success())

    def test_translation_drift_resets_after_success(self):
        self.stack()
        self.at(0)
        self.assertTrue(self.at(5).scalar_success())
        for name in PIECE_NAMES:
            self.data.qpos[self.model.joint(name + "_joint").qposadr[0]] += 0.004
        self.assertFalse(self.at(5.01).scalar_success())
        self.assertFalse(self.at(10).scalar_success())
        self.assertTrue(self.at(10.01).scalar_success())

    def test_rotation_drift_resets(self):
        self.stack()
        self.at(0)
        adr = self.model.joint("sms_cube_2_joint").qposadr[0]
        angle = np.deg2rad(12)
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        self.assertFalse(self.at(5).scalar_success())
        self.assertTrue(self.at(10).scalar_success())

    def test_floating_piece_and_touching_side_by_side_fail(self):
        self.stack()
        adr = self.model.joint("sms_roof_joint").qposadr[0]
        self.data.qpos[adr + 2] += 0.005
        self.at(0)
        self.assertFalse(self.at(10).scalar_success())
        self.evaluator.reset()
        for name, x, z in zip(
            PIECE_NAMES,
            [0.48, 0.5299, 0.5698, 0.6197],
            [0.7624, 0.7699, 0.7699, 0.7624],
        ):
            self.pose(name, [x, 0, z])
        self.assertEqual(self.at(0).metrics["num_blocks_in_tower"][0], 1)
        self.assertFalse(self.at(10).scalar_success())

    def test_order_is_not_prescribed(self):
        self.stack(("sms_cube_2", "sms_base", "sms_cube_1", "sms_roof"))
        self.at(0)
        self.assertTrue(self.at(5).scalar_success())

    def test_open_grippers_do_not_allow_an_externally_propped_tower(self):
        self.stack()
        wall = self.model.geom("back_wall_collision").id
        original = self.model.geom_pos[wall].copy()
        try:
            self.model.geom_pos[wall, 0] = 0.6349
            result = self.at(0)
            self.assertEqual(result.metrics["num_blocks_in_tower"][0], 4)
            self.assertFalse(result.metrics["released"][0])
            self.assertFalse(self.at(10).scalar_success())
        finally:
            self.model.geom_pos[wall] = original

    def test_reset_and_invalid_timestamps(self):
        self.stack()
        self.at(0)
        self.assertTrue(self.at(5).scalar_success())
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.at(0)
        self.evaluator.reset()
        self.assertFalse(self.at(0).scalar_success())
        with self.assertRaisesRegex(ValueError, "simulation time"):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None])
        with self.assertRaisesRegex(ValueError, "finite timestamp"):
            self.evaluator.evaluate_timed_qpos_batch(self.data.qpos[None], [np.nan])

    def test_batched_worlds_have_independent_physics_timers(self):
        self.stack()
        qpos = np.repeat(self.data.qpos[None], 2, axis=0)
        evaluator = make_task_evaluator(self.model, "stack_tallest_tower")
        evaluate_task_batch(evaluator, qpos, np.array([0, 10]))
        np.testing.assert_array_equal(
            evaluate_task_batch(evaluator, qpos, np.array([5, 14])).success,
            [True, False],
        )
        np.testing.assert_array_equal(
            evaluate_task_batch(evaluator, qpos, np.array([5, 15])).success,
            [True, True],
        )
        qpos[1, self.model.joint("right_left_finger").qposadr[0]] = 0
        np.testing.assert_array_equal(
            evaluate_task_batch(evaluator, qpos, np.array([6, 16])).success,
            [True, False],
        )

    def test_batched_env_passes_runtime_time_without_advancing_on_reads(self):
        self.stack()
        batch = BatchedWarpYAMEnv.__new__(BatchedWarpYAMEnv)
        batch._evaluator = self.evaluator
        batch.qpos = self.data.qpos[None]
        times = np.array([0.0])
        batch._runtime = SimpleNamespace(
            d_warp=SimpleNamespace(time=SimpleNamespace(numpy=lambda: times))
        )
        self.assertFalse(batch.evaluate().scalar_success())
        times[0] = 4.99
        for _ in range(20):
            self.assertFalse(batch.evaluate().scalar_success())
        times[0] = 5
        self.assertTrue(batch.evaluate().scalar_success())

    def test_pose_only_evaluators_remain_supported(self):
        result = TaskEvalResult(
            reward=np.array([0.25]), success=np.array([False]), metrics={}
        )
        evaluator = SimpleNamespace(evaluate_qpos_batch=lambda qpos: result)
        self.assertIs(
            evaluate_task_batch(evaluator, self.data.qpos[None], np.array([5])), result
        )

    def test_real_physics_stack_remains_stable_for_five_seconds(self):
        for dt in (0.002, 0.004):
            with self.subTest(physics_dt=dt):
                self.setUp()
                self.model.opt.timestep = dt
                self.stack()
                self.data.qvel[:] = 0
                mujoco.mj_forward(self.model, self.data)
                self.assertFalse(self.at(0).scalar_success())
                for _ in range(int(round(5 / dt))):
                    mujoco.mj_step(self.model, self.data)
                    result = self.env.evaluate_task()
                    if self.data.time < 5 - 1e-9:
                        self.assertFalse(result.scalar_success())
                self.assertTrue(result.scalar_success(), result.metrics)
        self.model.opt.timestep = 0.002


if __name__ == "__main__":
    unittest.main()
