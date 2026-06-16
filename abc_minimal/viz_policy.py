"""Live Viser viewer for an ABC-DiT sim rollout.

Visualise a policy in put bottles sim environment.
"""

from __future__ import annotations

import math
import threading
import time

import mujoco
import numpy as np
import torch
import viser
from mjviser import ViserMujocoScene

from abc_minimal.config import VizPolicyConfig, validate_model_config
from abc_minimal.eval_policy import (
    BOTTLE_COUNT,
    BOTTLE_Z,
    GRIPPER_CTRL_MAX,
    INIT_Q,
    TABLE_BOUNDS,
    PutBottlesEnv,
    SimPolicy,
    _flat_bottle_quat,
    _quat_mul,
    _quat_yaw,
    local_checkpoint,
    require_mjwarp,
    resolve_device,
)


def main(cfg: VizPolicyConfig) -> None:
    torch.set_float32_matmul_precision("high")

    errors = validate_model_config(cfg.sim.model)
    if errors:
        raise ValueError("Invalid sim eval config:\n  - " + "\n  - ".join(errors))

    require_mjwarp()
    ckpt_path = local_checkpoint(cfg.sim.checkpoint)
    device = resolve_device(cfg.sim.device)
    policy = SimPolicy(ckpt_path, cfg.sim, device)
    env = PutBottlesEnv(
        height=cfg.sim.camera_height,
        width=cfg.sim.camera_width,
        camera_keys=cfg.sim.model.camera_keys,
        prompt=cfg.sim.prompt,
        gpu_id=cfg.sim.gpu_id,
    )
    seed = [cfg.sim.seed]
    running = threading.Event()
    step_period_s = 1.0 / 30.0  # 30 Hz control rate

    def soft_reset(s: int) -> None:
        """Re-randomize bottle/bin positions without rebuilding the MjModel."""
        rng = np.random.default_rng(s)
        if env.model is None:
            env.reset(seed=s); return
        mujoco.mj_resetData(env.model, env.data)
        env._set_state(INIT_Q)
        bin_yaw = float(rng.uniform(-0.75, 0.75))
        bin_pos = [
            float(rng.uniform(0.57, 0.73)),
            float(rng.uniform(-0.25, 0.25)),
            0.83,
        ]
        bin_quat = _quat_mul(_quat_yaw(bin_yaw), np.array([0.70710678, 0.70710678, 0.0, 0.0]))
        env._set_freejoint("bin_joint", bin_pos, bin_quat.tolist())
        occupied = [(np.asarray(bin_pos[:2]), 0.13)]
        for index in range(BOTTLE_COUNT):
            for _ in range(200):
                x = float(rng.uniform(TABLE_BOUNDS[0], TABLE_BOUNDS[1]))
                y = float(rng.uniform(TABLE_BOUNDS[2], TABLE_BOUNDS[3]))
                if all(np.linalg.norm(np.array([x, y]) - c) > (0.055 + r + 0.04) for c, r in occupied):
                    break
            env._set_freejoint(
                f"bottle_{index + 1}_joint",
                [x, y, BOTTLE_Z],
                _flat_bottle_quat(float(rng.uniform(-math.pi, math.pi))).tolist(),
            )
            occupied.append((np.array([x, y]), 0.055))
        mujoco.mj_forward(env.model, env.data)
        env.sim.load_state(); env.sim.forward()
        env.evaluator.reset()

    def action_to_ctrl(action: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(env.model.nu, dtype=np.float32)
        for i, act_id in enumerate(env.ctrl_indices):
            v = float(action[i])
            if i in env.gripper_state_indices:
                v *= GRIPPER_CTRL_MAX
            ctrl[act_id] = v
        return ctrl

    def get_state_vanilla() -> np.ndarray:
        s = np.asarray(env.data.qpos[env.qpos_indices], dtype=np.float32)
        for i in env.gripper_state_indices:
            s[i] = float(np.clip(s[i] / GRIPPER_CTRL_MAX, 0.0, 1.0))
        return s

    def render_via_warp() -> dict[str, np.ndarray]:
        """Push env.data → d_warp and call mjwarp render once."""
        env.sim.load_state()
        env.sim.forward()
        rgb = env.sim.render()
        images = {}
        for name in cfg.sim.model.camera_keys:
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            images[name] = rgb[cam_id].transpose(2, 0, 1).copy()
        return images

    def obs_hybrid() -> dict:
        return {"state": get_state_vanilla(), "images": render_via_warp(), "prompt": cfg.sim.prompt}

    def evaluate_vanilla() -> dict:
        return env.evaluator.evaluate(np.asarray(env.data.qpos, dtype=np.float32))

    # Build initial scene.
    env.reset(seed=seed[0])
    if cfg.fast_inference:
        t0 = time.perf_counter()
        warmup_noise = np.random.default_rng(0).standard_normal(
            (cfg.sim.model.chunk_length, cfg.sim.model.action_dim), dtype=np.float32
        )
        policy.enable_fast_inference(
            compile_mode=cfg.fast_compile_mode,
            warmup_obs=obs_hybrid(),
            warmup_noise=warmup_noise,
        )
        torch.cuda.synchronize()
        print(f"fast inference ready in {time.perf_counter() - t0:.1f}s", flush=True)

    fid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fid >= 0:
        env.model.geom_rgba[fid, 3] = 0.0
    server = viser.ViserServer(host="0.0.0.0", port=cfg.port)
    status = server.gui.add_text("status", initial_value="idle", disabled=True)
    btn = server.gui.add_button("Reset & rollout")
    scene = ViserMujocoScene(server, env.model, num_envs=1)
    scene.update_from_mjdata(env.data)

    def rollout() -> None:
        print(f"[rollout] start seed={seed[0]}", flush=True)
        soft_reset(seed[0])
        obs = obs_hybrid()
        scene.update_from_mjdata(env.data)
        rng = np.random.default_rng(0)
        for chunk in range(cfg.sim.num_chunks):
            t_inf = time.perf_counter()
            actions = policy.infer(
                obs,
                noise=rng.standard_normal(
                    (cfg.sim.model.chunk_length, cfg.sim.model.action_dim), dtype=np.float32
                ),
            )
            t_after_infer = time.perf_counter()
            for action in actions[: cfg.sim.execute_chunk_dim]:
                t_step = time.perf_counter()
                env.data.ctrl[:] = action_to_ctrl(action)
                for _ in range(env.control_decimation):
                    mujoco.mj_step(env.model, env.data)
                scene.update_from_mjdata(env.data)
                ev = evaluate_vanilla()
                status.value = (
                    f"seed={seed[0]} chunk={chunk} bottles={ev['num_bottles_in_bin']}/4"
                )
                if ev["ever_success"]:
                    break
                # Realtime playback: sleep so each control step is ~33 ms wall-clock.
                sleep_s = step_period_s - (time.perf_counter() - t_step)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            t_obs = time.perf_counter()
            obs = obs_hybrid()
            t_end = time.perf_counter()
            print(
                f"chunk={chunk:2d} infer={(t_after_infer-t_inf)*1000:.0f}ms "
                f"steps={(t_obs-t_after_infer)*1000:.0f}ms "
                f"render={(t_end-t_obs)*1000:.0f}ms bottles={ev['num_bottles_in_bin']}",
                flush=True,
            )
            if evaluate_vanilla()["ever_success"]:
                break
        print(
            f"[rollout] done seed={seed[0]} bottles={evaluate_vanilla()['num_bottles_in_bin']}",
            flush=True,
        )
        status.value = (
            f"seed={seed[0]} done bottles={evaluate_vanilla()['num_bottles_in_bin']}/4"
        )
        running.clear()

    @btn.on_click
    def _(_) -> None:
        if running.is_set():
            print("[click] busy", flush=True); return
        seed[0] += 1
        running.set()
        threading.Thread(target=rollout, daemon=True).start()

    running.set()
    threading.Thread(target=rollout, daemon=True).start()
    print(f"viser ready on port {cfg.port}", flush=True)
    while True:
        time.sleep(60)
