"""Live Viser viewer for an ABC-DiT sim rollout.

Visualise a policy in put bottles sim environment.
"""

from __future__ import annotations

import concurrent.futures
import asyncio
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
    PutBottlesEnv,
    SimPolicy,
    _quat_mul,
    _quat_yaw,
    local_checkpoint,
    require_mjwarp,
    resolve_device,
    sample_bottle_pose,
    validate_rtc_config,
)


def main(cfg: VizPolicyConfig) -> None:
    torch.set_float32_matmul_precision("high")

    errors = validate_model_config(cfg.sim.model) + validate_rtc_config(cfg.sim)
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
        scene=cfg.sim.scene,
        gpu_id=cfg.sim.gpu_id,
    )
    seed = [cfg.sim.seed]
    running = threading.Event()
    reset_requested = threading.Event()
    step_period_s = 1.0 / 30.0  # 30 Hz control rate
    drag_lock = threading.Lock()
    drag_events: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    drag_initial_xy: dict[int, np.ndarray] = {}

    def clear_object_drags() -> None:
        with drag_lock:
            drag_events.clear()
            drag_initial_xy.clear()

    def apply_object_drags() -> None:
        if env.data is None:
            return
        with drag_lock:
            active_drags = {
                body_id: (start.copy(), end.copy())
                for body_id, (start, end) in drag_events.items()
            }
        max_xy_step = 0.01
        for body_id, (start, end) in active_drags.items():
            qpos_adr = draggable_qpos_addrs.get(body_id)
            qvel_adr = draggable_qvel_addrs.get(body_id)
            if qpos_adr is None or qvel_adr is None:
                continue
            with drag_lock:
                current_event = drag_events.get(body_id)
                if current_event is None or not np.array_equal(current_event[0], start):
                    continue
                initial_xy = drag_initial_xy.get(body_id)
                if initial_xy is None:
                    initial_xy = env.data.qpos[qpos_adr : qpos_adr + 2].copy()
                    drag_initial_xy[body_id] = initial_xy
                else:
                    initial_xy = initial_xy.copy()
            target_xy = initial_xy + end[:2] - start[:2]
            current_xy = env.data.qpos[qpos_adr : qpos_adr + 2]
            step_xy = target_xy - current_xy
            step_norm = float(np.linalg.norm(step_xy))
            if step_norm > max_xy_step:
                step_xy *= max_xy_step / step_norm
            current_xy += step_xy
            env.data.qvel[qvel_adr : qvel_adr + 2] = 0.0
            env.data.qvel[qvel_adr + 3 : qvel_adr + 6] *= 0.25

    def soft_reset(s: int) -> None:
        """Re-randomize bottle/bin positions without rebuilding the MjModel."""
        clear_object_drags()
        rng = np.random.default_rng(s)
        if env.model is None:
            env.reset(seed=s); return
        scene_config = env.scene
        mujoco.mj_resetData(env.model, env.data)
        env._set_state(np.asarray(scene_config.init_q, dtype=np.float32))
        bin_yaw = float(rng.uniform(*scene_config.bin_yaw_range))
        bin_scale = float(env.randomization.bin_scale)
        bin_x0, bin_x1, bin_y0, bin_y1 = scene_config.bin_xy_range
        bin_pos = [
            float(rng.uniform(bin_x0, bin_x1)),
            float(rng.uniform(bin_y0, bin_y1)),
            float(scene_config.bin_z_scale * bin_scale),
        ]
        bin_quat = _quat_mul(
            _quat_yaw(bin_yaw),
            np.asarray(scene_config.bin_base_quat, dtype=np.float64),
        )
        env._set_freejoint("bin_joint", bin_pos, bin_quat.tolist())
        occupied = [(np.asarray(bin_pos[:2]), scene_config.bin_occupied_radius * bin_scale)]
        for index in range(scene_config.bottle_count):
            scale = float(env.randomization.bottle_scales[index])
            pos, quat, center, radius = sample_bottle_pose(
                rng, scene_config, index, scale, occupied
            )
            env._set_freejoint(
                f"bottle_{index + 1}_joint",
                pos,
                quat.tolist(),
            )
            occupied.append((center, radius))
        mujoco.mj_forward(env.model, env.data)
        env.sim.load_state(); env.sim.forward()
        env.evaluator.reset()

    def action_to_ctrl(action: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(env.model.nu, dtype=np.float32)
        for i, act_id in enumerate(env.ctrl_indices):
            v = float(action[i])
            if i in env.gripper_state_indices:
                v *= env.scene.gripper_ctrl_max
            ctrl[act_id] = v
        return ctrl

    render_lock = threading.Lock()

    def state_from_qpos(qpos: np.ndarray) -> np.ndarray:
        s = np.asarray(qpos[env.qpos_indices], dtype=np.float32).copy()
        for i in env.gripper_state_indices:
            s[i] = float(np.clip(s[i] / env.scene.gripper_ctrl_max, 0.0, 1.0))
        return s

    def get_state_vanilla() -> np.ndarray:
        return state_from_qpos(np.asarray(env.data.qpos, dtype=np.float32))

    def snapshot_vanilla() -> dict[str, np.ndarray | float]:
        snap: dict[str, np.ndarray | float] = {
            "qpos": np.asarray(env.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float32).copy(),
            "ctrl": np.asarray(env.data.ctrl, dtype=np.float32).copy(),
            "time": float(env.data.time),
        }
        if env.model.na > 0:
            snap["act"] = np.asarray(env.data.act, dtype=np.float32).copy()
        if env.model.nmocap > 0:
            snap["mocap_pos"] = np.asarray(env.data.mocap_pos, dtype=np.float32).copy()
            snap["mocap_quat"] = np.asarray(env.data.mocap_quat, dtype=np.float32).copy()
        return snap

    def load_snapshot_to_warp(snapshot: dict[str, np.ndarray | float]) -> None:
        env.sim.mjw.reset_data(env.sim.m_warp, env.sim.d_warp)
        env.sim._copy(env.sim.d_warp.qpos, snapshot["qpos"][None], env.sim.wp.float32)
        if env.model.nv > 0:
            env.sim._copy(env.sim.d_warp.qvel, snapshot["qvel"][None], env.sim.wp.float32)
        if env.model.nu > 0:
            env.sim._copy(env.sim.d_warp.ctrl, snapshot["ctrl"][None], env.sim.wp.float32)
        if env.model.na > 0 and "act" in snapshot and hasattr(env.sim.d_warp, "act"):
            env.sim._copy(env.sim.d_warp.act, snapshot["act"][None], env.sim.wp.float32)
        if env.model.nmocap > 0 and "mocap_pos" in snapshot and "mocap_quat" in snapshot:
            env.sim._copy(env.sim.d_warp.mocap_pos, snapshot["mocap_pos"][None], env.sim.wp.vec3f)
            env.sim._copy(env.sim.d_warp.mocap_quat, snapshot["mocap_quat"][None], env.sim.wp.quatf)
        if hasattr(env.sim.d_warp, "time"):
            env.sim._copy(
                env.sim.d_warp.time,
                np.asarray([snapshot["time"]], dtype=np.float32),
                env.sim.wp.float32,
            )

    def render_via_warp() -> dict[str, np.ndarray]:
        """Push env.data → d_warp and call mjwarp render once."""
        with render_lock:
            env.sim.load_state()
            env.sim.forward()
            rgb = env.sim.render()
        return camera_images_from_rgb(rgb)

    def render_snapshot(snapshot: dict[str, np.ndarray | float]) -> dict[str, np.ndarray]:
        with render_lock:
            load_snapshot_to_warp(snapshot)
            env.sim.forward()
            rgb = env.sim.render()
        return camera_images_from_rgb(rgb)

    def camera_images_from_rgb(rgb: np.ndarray) -> dict[str, np.ndarray]:
        images = {}
        for name in cfg.sim.model.camera_keys:
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            images[name] = rgb[cam_id].transpose(2, 0, 1).copy()
        return images

    def obs_hybrid() -> dict:
        return {"state": get_state_vanilla(), "images": render_via_warp(), "prompt": cfg.sim.prompt}

    def obs_from_snapshot(snapshot: dict[str, np.ndarray | float]) -> dict:
        return {
            "state": state_from_qpos(snapshot["qpos"]),
            "images": render_snapshot(snapshot),
            "prompt": cfg.sim.prompt,
        }

    def evaluate_vanilla() -> dict:
        return env.evaluator.evaluate(np.asarray(env.data.qpos, dtype=np.float32))

    class _AsyncRTCManager:
        def __init__(self):
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._pending: concurrent.futures.Future[tuple[np.ndarray, float, float]] | None = None

        def _action_prefix(self, current_actions: np.ndarray) -> np.ndarray:
            m = cfg.sim.model
            prefix = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
            executed = np.asarray(current_actions[: cfg.sim.execute_chunk_dim], dtype=np.float32)
            prefix[: cfg.sim.rtc_prefix_length] = executed[-cfg.sim.rtc_prefix_length :]
            return prefix

        def start(
            self,
            snapshot: dict[str, np.ndarray | float],
            current_actions: np.ndarray,
            noise: np.ndarray,
        ) -> None:
            if self._pending is not None:
                raise RuntimeError("RTC inference is already pending")
            action_prefix = self._action_prefix(current_actions)

            def _run() -> tuple[np.ndarray, float, float]:
                t_obs = time.perf_counter()
                next_obs = obs_from_snapshot(snapshot)
                obs_s = time.perf_counter() - t_obs
                t_inf = time.perf_counter()
                next_actions = policy.infer(
                    next_obs,
                    noise=noise,
                    action_prefix=action_prefix,
                    prefix_length=cfg.sim.rtc_prefix_length,
                )
                infer_s = time.perf_counter() - t_inf
                return next_actions[cfg.sim.rtc_prefix_length :], obs_s, infer_s

            self._pending = self._executor.submit(_run)

        def get(self) -> tuple[np.ndarray, float, float, bool]:
            if self._pending is None:
                raise RuntimeError("No RTC inference is pending")
            ready = self._pending.done()
            actions, obs_s, infer_s = self._pending.result()
            self._pending = None
            return actions, obs_s, infer_s, ready

        def close(self) -> None:
            if self._pending is not None:
                self._pending.result()
                self._pending = None
            self._executor.shutdown(wait=True)

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
    if cfg.sim.rtc:
        t0 = time.perf_counter()
        warmup_noise = np.random.default_rng(0).standard_normal(
            (cfg.sim.model.chunk_length, cfg.sim.model.action_dim), dtype=np.float32
        )
        policy.warmup_rtc(obs_hybrid(), warmup_noise, cfg.sim.rtc_prefix_length)
        print(f"rtc inference ready in {time.perf_counter() - t0:.1f}s", flush=True)

    fid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fid >= 0:
        env.model.geom_rgba[fid, 3] = 0.0
    server = viser.ViserServer(host="0.0.0.0", port=cfg.port)
    actual_port = server.get_port()

    default_camera_position = np.array([-0.42, 0.0, 1.66], dtype=np.float64)
    default_camera_look_at = np.array([0.45, 0.0, 0.87], dtype=np.float64)

    def apply_default_view(client: viser.ClientHandle) -> None:
        client.camera.position = default_camera_position
        client.camera.look_at = default_camera_look_at
        client.camera.up_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        client.camera.fov = math.radians(50.0)

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        await asyncio.sleep(0.1)
        apply_default_view(client)

    status = server.gui.add_text("status", initial_value="idle", disabled=True)
    btn = server.gui.add_button("Reset & rollout")
    view_btn = server.gui.add_button("Default view")
    drag_enabled = server.gui.add_checkbox("Object drag", initial_value=True)
    scene = ViserMujocoScene(server, env.model, num_envs=1)
    scene.camera_tracking_enabled = False
    server.scene.remove_by_name("/fixed_bodies/world/table_plane")
    scene.update_from_mjdata(env.data)

    draggable_qpos_addrs: dict[int, int] = {}
    draggable_qvel_addrs: dict[int, int] = {}
    draggable_root_ids = set()
    for name in env.evaluator.bottle_names:
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        if body_id < 0 or joint_id < 0:
            continue
        draggable_root_ids.add(body_id)
        draggable_qpos_addrs[body_id] = int(env.model.jnt_qposadr[joint_id])
        draggable_qvel_addrs[body_id] = int(env.model.jnt_dofadr[joint_id])
    bin_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
    bin_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
    if bin_body_id >= 0 and bin_joint_id >= 0:
        draggable_root_ids.add(bin_body_id)
        draggable_qpos_addrs[bin_body_id] = int(env.model.jnt_qposadr[bin_joint_id])
        draggable_qvel_addrs[bin_body_id] = int(env.model.jnt_dofadr[bin_joint_id])

    def draggable_root_for_body(body_id: int) -> int | None:
        while body_id > 0:
            if body_id in draggable_root_ids:
                return body_id
            body_id = int(env.model.body_parentid[body_id])
        return None

    def register_drag_callbacks() -> None:
        num_callbacks = 0
        for mesh_group in scene._mesh_groups:
            if int(mesh_group.group_id) != 2:
                continue
            target_body_ids = [draggable_root_for_body(int(body_id)) for body_id in mesh_group.body_ids]
            if not any(body_id is not None for body_id in target_body_ids):
                continue

            @mesh_group.handle.on_drag
            async def _drag(event, *, body_ids=target_body_ids) -> None:
                if event.instance_index is None:
                    return
                target_body_id = body_ids[int(event.instance_index) % len(body_ids)]
                if target_body_id is None:
                    return
                if event.phase == "end" or not drag_enabled.value:
                    with drag_lock:
                        drag_events.pop(target_body_id, None)
                        drag_initial_xy.pop(target_body_id, None)
                    return

                start = np.asarray(event.start_position, dtype=np.float64)
                end = np.asarray(event.end_position, dtype=np.float64)
                with drag_lock:
                    drag_events[target_body_id] = (start, end)

            num_callbacks += 1
        print(f"registered object drag callbacks on {num_callbacks} viser mesh handles", flush=True)

    register_drag_callbacks()

    @view_btn.on_click
    def _(_) -> None:
        for client in server.get_clients().values():
            apply_default_view(client)

    def rollout() -> None:
        rollout_seed = seed[0]
        print(f"[rollout] start seed={rollout_seed}", flush=True)
        soft_reset(rollout_seed)
        obs = obs_hybrid()
        scene.update_from_mjdata(env.data)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(
            (cfg.sim.model.chunk_length, cfg.sim.model.action_dim), dtype=np.float32
        )
        t_inf = time.perf_counter()
        actions = policy.infer(obs, noise=noise)
        current_infer_s = time.perf_counter() - t_inf
        rtc = _AsyncRTCManager() if cfg.sim.rtc else None
        cancelled = False
        try:
            for chunk in range(cfg.sim.num_chunks):
                if reset_requested.is_set():
                    cancelled = True
                    break
                t_chunk = time.perf_counter()
                chunk_infer_s = current_infer_s
                t_steps = time.perf_counter()
                rtc_started = False
                rtc_ready = None
                rtc_infer_s = None
                obs_render_s = 0.0
                lead_index = cfg.sim.execute_chunk_dim - cfg.sim.rtc_inference_lead_steps
                for action_index, action in enumerate(actions[: cfg.sim.execute_chunk_dim]):
                    if reset_requested.is_set():
                        cancelled = True
                        break
                    t_step = time.perf_counter()
                    if (
                        rtc is not None
                        and chunk + 1 < cfg.sim.num_chunks
                        and action_index == lead_index
                    ):
                        snapshot = snapshot_vanilla()
                        next_noise = rng.standard_normal(
                            (cfg.sim.model.chunk_length, cfg.sim.model.action_dim),
                            dtype=np.float32,
                        )
                        rtc.start(snapshot, actions, next_noise)
                        rtc_started = True
                    env.data.ctrl[:] = action_to_ctrl(action)
                    for _ in range(env.control_decimation):
                        apply_object_drags()
                        mujoco.mj_step(env.model, env.data)
                    scene.update_from_mjdata(env.data)
                    ev = evaluate_vanilla()
                    status.value = (
                        f"seed={rollout_seed} chunk={chunk} bottles={ev['num_bottles_in_bin']}/4"
                    )
                    if ev["ever_success"]:
                        break
                    # Realtime playback: sleep so each control step is ~33 ms wall-clock.
                    sleep_s = step_period_s - (time.perf_counter() - t_step)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                steps_s = time.perf_counter() - t_steps
                if cancelled:
                    break
                if ev["ever_success"]:
                    break
                if rtc is None and chunk + 1 < cfg.sim.num_chunks:
                    t_obs = time.perf_counter()
                    obs = obs_hybrid()
                    obs_render_s = time.perf_counter() - t_obs
                    noise = rng.standard_normal(
                        (cfg.sim.model.chunk_length, cfg.sim.model.action_dim),
                        dtype=np.float32,
                    )
                    t_inf = time.perf_counter()
                    actions = policy.infer(obs, noise=noise)
                    current_infer_s = time.perf_counter() - t_inf
                elif rtc_started:
                    actions, obs_render_s, rtc_infer_s, rtc_ready = rtc.get()
                    current_infer_s = rtc_infer_s
                logged_infer_s = rtc_infer_s if rtc_infer_s is not None else chunk_infer_s
                print(
                    f"chunk={chunk:2d} infer={logged_infer_s*1000:.0f}ms "
                    f"steps={steps_s*1000:.0f}ms "
                    f"render={obs_render_s*1000:.0f}ms bottles={ev['num_bottles_in_bin']} "
                    f"rtc_ready_at_chunk_end={rtc_ready}",
                    flush=True,
                )
                if evaluate_vanilla()["ever_success"]:
                    break
        finally:
            if rtc is not None:
                rtc.close()
        final_eval = evaluate_vanilla()
        bottles = final_eval["num_bottles_in_bin"]
        success = bool(final_eval["ever_success"])
        if cancelled:
            print(f"[rollout] reset seed={rollout_seed} bottles={bottles}", flush=True)
            status.value = f"seed={rollout_seed} reset"
        else:
            print(f"[rollout] done seed={rollout_seed} bottles={bottles}", flush=True)
            status.value = f"seed={rollout_seed} done bottles={bottles}/4"
            if success and not reset_requested.is_set():
                deadline = time.perf_counter() + 3.0
                while not reset_requested.is_set() and time.perf_counter() < deadline:
                    time.sleep(0.05)
        if reset_requested.is_set():
            reset_requested.clear()
            threading.Thread(target=rollout, daemon=True).start()
        else:
            seed[0] += 1
            print(f"[rollout] auto reset seed={seed[0]}", flush=True)
            status.value = f"resetting to seed={seed[0]}"
            threading.Thread(target=rollout, daemon=True).start()

    @btn.on_click
    def _(_) -> None:
        if running.is_set():
            seed[0] += 1
            print(f"[click] reset seed={seed[0]}", flush=True)
            status.value = f"resetting to seed={seed[0]}"
            reset_requested.set()
            return
        seed[0] += 1
        reset_requested.clear()
        running.set()
        threading.Thread(target=rollout, daemon=True).start()

    running.set()
    threading.Thread(target=rollout, daemon=True).start()
    print(f"Viser ready on port {actual_port}", flush=True)
    while True:
        time.sleep(60)
