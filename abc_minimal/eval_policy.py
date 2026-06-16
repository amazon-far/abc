"""Run MuJoCo-Warp put-bottles eval for ABC-DiT checkpoints.

Builds the scene, executes policy rollouts, and writes JSON/video outputs.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from abc_minimal.config import SimEvalConfig, validate_model_config
from abc_minimal.dit import (
    CLIPTextEmbedder,
    DiTPolicy,
    load_pretrained,
)
from abc_minimal.preprocess import normalize, parse_norm_stats, resize_pad_normalize, unnormalize

torch.set_float32_matmul_precision("high")


# Config.

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "assets" / "put_bottles" / "put_bottle.xml"
GRIPPER_CTRL_MAX = 0.0475
BOTTLE_COUNT = 4
INIT_Q = np.array(
    [0, 1.047, 1.047, 0, 0, 0, 0, 0, 1.047, 1.047, 0, 0, 0, 0],
    dtype=np.float32,
)
BOTTLE_Z = 0.754
TABLE_BOUNDS = (0.34, 0.86, -0.52, 0.52)


# XML helpers.


def _fmt(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _flat_bottle_quat(yaw: float) -> np.ndarray:
    flat = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0], dtype=np.float64)
    q = _quat_mul(_quat_yaw(yaw), flat)
    return q / np.linalg.norm(q)


def scene_xml(bottle_scales: np.ndarray, bin_scale: float) -> str:
    root = ET.fromstring(SCENE_XML.read_text())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
        compiler.set("texturedir", str((ROOT / "assets" / "put_bottles" / "assets").resolve()))
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name", "")
        scale = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()], dtype=np.float64)
        for idx in range(BOTTLE_COUNT):
            if name.startswith(f"bottle_{idx}_"):
                mesh.set("scale", _fmt(scale * float(bottle_scales[idx])))
                break
        if name.startswith("water_bottle_"):
            mesh.set("scale", _fmt(scale * float(bin_scale)))
    return ET.tostring(root, encoding="unicode")


# Environment and metrics.


class PutBottlesEvaluator:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.bottle_names, self.bottle_qpos_addrs = self._bottle_addrs()
        self.bin_qpos_adr = self._joint_qpos_adr("bin_joint")
        self.max_bottles = 0
        self.ever_success = False

    def reset(self) -> None:
        self.max_bottles = 0
        self.ever_success = False

    def _joint_qpos_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _bottle_addrs(self) -> tuple[list[str], np.ndarray]:
        entries = []
        for joint_id in range(self.model.njnt):
            name = self.model.jnt(joint_id).name
            match = re.fullmatch(r"bottle_(\d+)_joint", name or "")
            if match:
                idx = int(match.group(1))
                entries.append((idx, f"bottle_{idx}", int(self.model.jnt_qposadr[joint_id])))
        entries.sort()
        return [e[1] for e in entries], np.asarray([e[2] for e in entries], dtype=np.int32)

    def evaluate(self, qpos: np.ndarray) -> dict[str, Any]:
        qpos = np.asarray(qpos, dtype=np.float32)
        bin_pos = qpos[self.bin_qpos_adr : self.bin_qpos_adr + 3]
        bottle_pos = np.stack([qpos[adr : adr + 3] for adr in self.bottle_qpos_addrs])
        rel = bottle_pos - bin_pos[None]
        radial = np.linalg.norm(rel[:, :2], axis=1)
        radial_margin = 0.155 - radial
        lower_height_margin = rel[:, 2] + 0.06
        upper_height_margin = 0.26 - rel[:, 2]
        in_bin = (radial_margin >= 0.0) & (lower_height_margin >= 0.0) & (upper_height_margin >= 0.0)
        num = int(in_bin.sum())
        active = len(self.bottle_names)
        self.max_bottles = max(self.max_bottles, num)
        success = num == active
        self.ever_success = self.ever_success or success
        return {
            "reward": float(num / max(active, 1)),
            "success": bool(success),
            "ever_success": bool(self.ever_success),
            "num_bottles_in_bin": num,
            "num_active_bottles": active,
            "max_bottles_in_bin_so_far": int(self.max_bottles),
            "bottle_in_bin_mask": [bool(x) for x in in_bin.tolist()],
            "bottles_in_bin": [name for name, ok in zip(self.bottle_names, in_bin) if bool(ok)],
            "bottle_names": list(self.bottle_names),
            "success_count": active,
            "closest_radial_margin": float(radial_margin.max()),
            "closest_height_margin": float(lower_height_margin.max()),
            "closest_upper_height_margin": float(upper_height_margin.max()),
        }


@dataclass
class Randomization:
    seed: int
    bottle_states: dict[str, dict[str, list[float]]]
    bin_state: dict[str, list[float]]
    bottle_scales: list[float]
    bin_scale: float


def require_mjwarp() -> None:
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "eval_policy.py uses MJWarp only. Install/import `mujoco_warp` and `warp` "
            "in the eval environment before running sim eval."
        ) from exc


class MJWarpSim:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, height: int, width: int, gpu_id: int | None):
        require_mjwarp()
        import mujoco_warp as mjw
        import warp as wp

        self.mjw = mjw
        self.wp = wp
        self.model = model
        self.data = data
        self.height = height
        self.width = width
        self.nworld = 1
        if gpu_id is not None:
            wp.set_device(f"cuda:{gpu_id}")
        self.m_warp = mjw.put_model(model)
        self.d_warp = mjw.put_data(model, data, nworld=self.nworld, nconmax=model.nconmax, njmax=model.njmax)
        self.render_context = mjw.create_render_context(
            mjm=model,
            nworld=self.nworld,
            cam_res=(width, height),
            render_rgb=[True] * model.ncam,
            render_depth=[False] * model.ncam,
            use_textures=True,
            use_shadows=True,
        )
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.wp.synchronize()
        except Exception:
            pass
        self.render_context = None
        self.m_warp = None
        self.d_warp = None
        self.model = None
        self.data = None
        self.closed = True

    def _copy(self, target: Any, values: np.ndarray, dtype: Any) -> None:
        self.wp.copy(target, self.wp.from_numpy(np.asarray(values), dtype=dtype))

    def load_state(self) -> None:
        self.mjw.reset_data(self.m_warp, self.d_warp)
        self._copy(self.d_warp.qpos, np.asarray(self.data.qpos, dtype=np.float32)[None], self.wp.float32)
        if self.model.nv > 0:
            self._copy(self.d_warp.qvel, np.asarray(self.data.qvel, dtype=np.float32)[None], self.wp.float32)
        if self.model.nu > 0:
            self._copy(self.d_warp.ctrl, np.asarray(self.data.ctrl, dtype=np.float32)[None], self.wp.float32)
        if self.model.na > 0 and hasattr(self.d_warp, "act"):
            self._copy(self.d_warp.act, np.asarray(self.data.act, dtype=np.float32)[None], self.wp.float32)
        if self.model.nmocap > 0:
            self._copy(self.d_warp.mocap_pos, np.asarray(self.data.mocap_pos, dtype=np.float32)[None], self.wp.vec3f)
            self._copy(self.d_warp.mocap_quat, np.asarray(self.data.mocap_quat, dtype=np.float32)[None], self.wp.quatf)
        if hasattr(self.d_warp, "time"):
            self._copy(self.d_warp.time, np.asarray([self.data.time], dtype=np.float32), self.wp.float32)

    def forward(self) -> None:
        self.mjw.forward(self.m_warp, self.d_warp)

    def qpos(self) -> np.ndarray:
        return self.d_warp.qpos.numpy()[0].copy()

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float32)
        if ctrl.shape != (self.model.nu,):
            raise ValueError(f"Expected ctrl shape {(self.model.nu,)}, got {ctrl.shape}")
        self._copy(self.d_warp.ctrl, ctrl[None], self.wp.float32)

    def step(self, nstep: int) -> None:
        for _ in range(nstep):
            self.mjw.step(self.m_warp, self.d_warp)

    def render(self) -> np.ndarray:
        self.mjw.refit_bvh(self.m_warp, self.d_warp, self.render_context)
        self.mjw.render(self.m_warp, self.d_warp, self.render_context)
        rgba = self.render_context.rgb_data.numpy().view(np.uint8).reshape(
            self.nworld,
            self.model.ncam,
            self.height,
            self.width,
            4,
        )
        return rgba[0, :, :, :, :3][..., ::-1].copy()


class PutBottlesEnv:
    def __init__(
        self,
        *,
        height: int,
        width: int,
        camera_keys: tuple[str, ...],
        prompt: str,
        control_decimation: int = 17,
        gpu_id: int | None = None,
    ):
        self.height = height
        self.width = width
        self.camera_keys = tuple(camera_keys)
        self.prompt = prompt
        self.control_decimation = control_decimation
        self.gpu_id = gpu_id
        self.sim = None
        self.model = None
        self.data = None
        self.evaluator = None
        self.qpos_indices: list[int] = []
        self.ctrl_indices: list[int] = []
        self.gripper_state_indices: set[int] = set()
        self.randomization = None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def _bind(self, xml: str) -> None:
        self.close()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.sim = MJWarpSim(self.model, self.data, height=self.height, width=self.width, gpu_id=self.gpu_id)
        self.evaluator = PutBottlesEvaluator(self.model)
        self.qpos_indices, self.ctrl_indices, self.gripper_state_indices = [], [], set()
        idx = 0
        for robot in ("left", "right"):
            for j in range(1, 7):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_joint{j}")
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_joint{j}")
                self.qpos_indices.append(int(self.model.jnt_qposadr[jid]))
                self.ctrl_indices.append(aid)
                idx += 1
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_left_finger")
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_gripper")
            self.qpos_indices.append(int(self.model.jnt_qposadr[jid]))
            self.ctrl_indices.append(aid)
            self.gripper_state_indices.add(idx)
            idx += 1

    def reset(self, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        bottle_scales = rng.uniform(0.9, 1.1, size=BOTTLE_COUNT).astype(np.float32)
        bin_scale = float(rng.uniform(0.95, 1.05))
        self._bind(scene_xml(bottle_scales, bin_scale))
        mujoco.mj_resetData(self.model, self.data)
        self._set_state(INIT_Q)

        bin_yaw = float(rng.uniform(-0.75, 0.75))
        bin_pos = [float(rng.uniform(0.57, 0.73)), float(rng.uniform(-0.25, 0.25)), float(0.83 * bin_scale)]
        bin_quat = _quat_mul(_quat_yaw(bin_yaw), np.array([0.70710678, 0.70710678, 0.0, 0.0]))
        self._set_freejoint("bin_joint", bin_pos, bin_quat.tolist())

        occupied: list[tuple[np.ndarray, float]] = [(np.asarray(bin_pos[:2]), 0.13 * bin_scale)]
        bottle_states = {}
        for index in range(BOTTLE_COUNT):
            for _ in range(200):
                x = float(rng.uniform(TABLE_BOUNDS[0], TABLE_BOUNDS[1]))
                y = float(rng.uniform(TABLE_BOUNDS[2], TABLE_BOUNDS[3]))
                radius = 0.055 * float(bottle_scales[index])
                if all(np.linalg.norm(np.array([x, y]) - c) > (radius + r + 0.04) for c, r in occupied):
                    break
            yaw = float(rng.uniform(-math.pi, math.pi))
            q = _flat_bottle_quat(yaw)
            pos = [x, y, BOTTLE_Z]
            name = f"bottle_{index + 1}_joint"
            self._set_freejoint(name, pos, q.tolist())
            bottle_states[name] = {"pos": pos, "quat": q.tolist(), "scale": [float(bottle_scales[index])]}
            occupied.append((np.array([x, y]), radius))

        mujoco.mj_forward(self.model, self.data)
        self.sim.load_state()
        self.sim.forward()
        self.evaluator.reset()
        self.randomization = Randomization(
            seed=seed,
            bottle_states=bottle_states,
            bin_state={"pos": bin_pos, "quat": bin_quat.tolist(), "yaw": [bin_yaw]},
            bottle_scales=bottle_scales.tolist(),
            bin_scale=bin_scale,
        )
        return self.obs()

    def _set_freejoint(self, name: str, pos: list[float], quat: list[float]) -> None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(self.model.jnt_qposadr[jid])
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat

    def _set_state(self, state: np.ndarray) -> None:
        for i, qpos_idx in enumerate(self.qpos_indices):
            val = float(state[i])
            if i in self.gripper_state_indices:
                val *= GRIPPER_CTRL_MAX
            self.data.qpos[qpos_idx] = val

    def get_state(self) -> np.ndarray:
        state = np.asarray(self.sim.qpos()[self.qpos_indices], dtype=np.float32)
        for i in self.gripper_state_indices:
            state[i] = np.clip(state[i] / GRIPPER_CTRL_MAX, 0.0, 1.0)
        return state

    def render_cameras(self) -> dict[str, np.ndarray]:
        rgb = self.sim.render()
        images = {}
        for name in self.camera_keys:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id < 0:
                raise ValueError(f"Camera not found: {name}")
            images[name] = rgb[cam_id].transpose(2, 0, 1).copy()
        return images

    def obs(self) -> dict[str, Any]:
        return {"state": self.get_state(), "images": self.render_cameras(), "prompt": self.prompt}

    def step_one(self, action: np.ndarray) -> None:
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for i, act_id in enumerate(self.ctrl_indices):
            val = float(action[i])
            if i in self.gripper_state_indices:
                val *= GRIPPER_CTRL_MAX
            ctrl[act_id] = val
        self.sim.set_ctrl(ctrl)
        self.sim.step(self.control_decimation)

    def evaluate(self) -> dict[str, Any]:
        return self.evaluator.evaluate(self.sim.qpos())


# Policy adapter.


def load_json_or_s3(path: str) -> dict[str, Any]:
    if path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            subprocess.run(["aws", "s3", "cp", path, tmp.name], check=True)
            return json.loads(Path(tmp.name).read_text())
    return json.loads(Path(path).expanduser().read_text())


def local_checkpoint(path: str) -> Path:
    if not path.startswith("s3://"):
        return Path(path).expanduser().resolve()
    out_dir = ROOT / "checkpoints" / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / Path(path).name
    if not out.exists():
        subprocess.run(["aws", "s3", "cp", path, str(out)], check=True)
    return out


def resolve_norm_stats(ckpt: dict[str, Any], override: str | None) -> dict[str, Any]:
    if override:
        raw = load_json_or_s3(override)
    elif ckpt.get("norm_stats") is not None:
        raw = ckpt["norm_stats"]
    else:
        raise ValueError("No norm_stats in checkpoint; pass --norm-stats-path")
    return parse_norm_stats(raw)


class _FastInferenceGraph:
    """Fixed-shape outer CUDA graph over `model.sample_actions`.

    Captures one bf16 batch on GPU; subsequent .infer() calls just memcpy
    fresh inputs into the static tensors and replay the graph.
    """
    def __init__(self, policy: "SimPolicy"):
        self.policy = policy
        self.model = policy.model
        self.device = policy.device
        self.dtype = torch.bfloat16
        self.graph = torch.cuda.CUDAGraph()
        self.output: torch.Tensor | None = None
        m = policy.config.model

        self.static_state = torch.empty(1, m.state_dim, device=self.device, dtype=self.dtype)
        self.static_noise = torch.empty(
            1, m.chunk_length, m.action_dim, device=self.device, dtype=self.dtype
        )
        self.static_images = {
            cam: torch.empty(1, 3, 224, 224, device=self.device, dtype=self.dtype)
            for cam in m.camera_keys
        }
        self.static_task_vec = policy.task_vec.to(
            device=self.device, dtype=self.dtype
        ).clone()
        self.batch = {
            "state": self.static_state,
            "actions": torch.zeros(
                1, m.chunk_length, m.action_dim, device=self.device, dtype=self.dtype
            ),
            "images": self.static_images,
            "task_vec_clip": self.static_task_vec,
        }

    def _copy_inputs(self, obs: dict[str, Any], noise: np.ndarray | None) -> None:
        m = self.policy.config.model
        state = normalize(
            np.asarray(obs["state"], dtype=np.float32), self.policy.norm_stats["state"]
        )
        self.static_state.copy_(
            torch.from_numpy(state[None]).to(device=self.device, dtype=self.dtype)
        )
        if noise is None:
            self.static_noise.normal_()
        else:
            noise_arr = noise[None].astype(np.float32, copy=False)
            if noise_arr.shape != (1, m.chunk_length, m.action_dim):
                raise ValueError(
                    f"fast inference expects noise shape "
                    f"{(m.chunk_length, m.action_dim)}, got {noise.shape}"
                )
            self.static_noise.copy_(
                torch.from_numpy(noise_arr).to(device=self.device, dtype=self.dtype)
            )
        for cam in m.camera_keys:
            self.static_images[cam].copy_(
                resize_pad_normalize(obs["images"][cam])
                .unsqueeze(0)
                .to(device=self.device, dtype=self.dtype)
            )

    def capture(
        self,
        warmup_obs: dict[str, Any],
        warmup_noise: np.ndarray | None,
        replay_warmups: int,
    ) -> None:
        self._copy_inputs(warmup_obs, warmup_noise)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                self.output = self.model.sample_actions(
                    self.batch,
                    num_steps=self.policy.diffusion_steps,
                    noise=self.static_noise,
                )
        torch.cuda.current_stream().wait_stream(stream)

        with torch.cuda.graph(self.graph):
            self.output = self.model.sample_actions(
                self.batch, num_steps=self.policy.diffusion_steps, noise=self.static_noise
            )

        for _ in range(replay_warmups):
            self._copy_inputs(warmup_obs, warmup_noise)
            self.graph.replay()
            assert self.output is not None
            _ = self.output[0].float().detach().cpu().numpy()
        torch.cuda.synchronize()

    def infer(self, obs: dict[str, Any], noise: np.ndarray | None) -> np.ndarray:
        self._copy_inputs(obs, noise)
        self.graph.replay()
        assert self.output is not None
        actions_np = self.output[0].float().detach().cpu().numpy()
        return unnormalize(actions_np, self.policy.norm_stats["actions"]).astype(np.float32)


class SimPolicy:
    def __init__(self, checkpoint: Path, config: SimEvalConfig, device: str):
        self.config = config
        self.device = torch.device(device)
        self.diffusion_steps = config.diffusion_steps
        self.model = DiTPolicy(config.model).to(self.device)
        ckpt = load_pretrained(self.model, checkpoint)
        self.model.eval()
        self.norm_stats = resolve_norm_stats(ckpt, config.norm_stats_path)
        self.embedder = CLIPTextEmbedder(config.clip, device=self.device)
        self.task_vec = self.embedder.encode([config.prompt]).to(self.device)
        self._fast_graph: _FastInferenceGraph | None = None

    def enable_fast_inference(
        self,
        compile_mode: str = "max-autotune-no-cudagraphs",
        replay_warmups: int = 24,
        warmup_obs: dict[str, Any] | None = None,
        warmup_noise: np.ndarray | None = None,
    ) -> None:
        """One-time setup for ~4-5x faster inference on H100/H200.

        Casts the model to bf16, compiles `predict_velocity`, and captures
        `sample_actions` in an outer CUDA graph. Re-call .infer() afterwards
        and it transparently replays the graph (~41 ms vs ~190 ms eager).
        """
        if self._fast_graph is not None:
            return
        if self.device.type != "cuda":
            raise RuntimeError("fast inference requires a CUDA device")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.model.to(torch.bfloat16)
        self.model.img_backbone.set_bfloat16(True)
        self.task_vec = self.task_vec.to(device=self.device, dtype=torch.bfloat16)

        compile_kwargs: dict[str, Any] = {"dynamic": False}
        if compile_mode:
            compile_kwargs["mode"] = compile_mode
        self.model.predict_velocity = torch.compile(
            self.model.predict_velocity, **compile_kwargs
        )

        m = self.config.model
        if warmup_obs is None:
            warmup_obs = {
                "state": np.zeros(m.state_dim, dtype=np.float32),
                "images": {
                    cam: np.zeros((3, 168, 224), dtype=np.uint8) for cam in m.camera_keys
                },
                "prompt": self.config.prompt,
            }
        if warmup_noise is None:
            warmup_noise = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
        self._fast_graph = _FastInferenceGraph(self)
        self._fast_graph.capture(warmup_obs, warmup_noise, replay_warmups)

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], noise: np.ndarray | None = None) -> np.ndarray:
        if self._fast_graph is not None:
            return self._fast_graph.infer(obs, noise)
        state = normalize(np.asarray(obs["state"], dtype=np.float32), self.norm_stats["state"])
        batch = {
            "state": torch.from_numpy(state[None]).float().to(self.device),
            "actions": torch.zeros(
                1, self.config.model.chunk_length, self.config.model.action_dim, device=self.device
            ),
            "images": {
                cam: resize_pad_normalize(obs["images"][cam]).unsqueeze(0).to(self.device)
                for cam in self.config.model.camera_keys
            },
            "task_vec_clip": self.task_vec,
        }
        noise_t = None
        if noise is not None:
            noise_t = torch.from_numpy(noise[None].astype(np.float32)).to(self.device)
        actions = self.model.sample_actions(batch, num_steps=self.diffusion_steps, noise=noise_t)
        actions_np = actions[0].detach().cpu().numpy()
        return unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)


# Rollout.


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if hasattr(x, "__dataclass_fields__"):
        return jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def video_frame(images: dict[str, np.ndarray], camera_keys: tuple[str, ...]) -> np.ndarray:
    frames = []
    for name in camera_keys:
        frame = images[name].transpose(1, 2, 0)
        frames.append(np.ascontiguousarray(frame))
    return np.concatenate(frames, axis=1)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_eval(config: SimEvalConfig) -> dict[str, Any]:
    model_errors = validate_model_config(config.model)
    if model_errors:
        raise ValueError("Invalid sim eval config:\n  - " + "\n  - ".join(model_errors))

    require_mjwarp()
    ckpt_path = local_checkpoint(config.checkpoint)
    device = resolve_device(config.device)
    policy = SimPolicy(ckpt_path, config, device)
    env = PutBottlesEnv(
        height=config.camera_height,
        width=config.camera_width,
        camera_keys=config.model.camera_keys,
        prompt=config.prompt,
        gpu_id=config.gpu_id,
    )
    rng = np.random.default_rng(config.policy_seed)
    worlds = []
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for world_index in range(config.num_worlds):
            video = None
            video_path = None
            t0 = time.perf_counter()
            seed = int(config.seed + world_index)
            obs = env.reset(seed=seed)
            if config.save_video:
                import imageio.v2 as imageio

                video_path = out_dir / f"world_{world_index:03d}.mp4"
                video = imageio.get_writer(str(video_path), fps=config.video_fps, macro_block_size=1)
                video.append_data(video_frame(obs["images"], config.model.camera_keys))
            final_eval = env.evaluate()
            steps = 0
            try:
                for chunk in range(config.num_chunks):
                    noise = rng.standard_normal(
                        (config.model.chunk_length, config.model.action_dim), dtype=np.float32
                    )
                    actions = policy.infer(obs, noise=noise)
                    for action in actions[: config.execute_chunk_dim]:
                        env.step_one(action)
                        steps += 1
                        final_eval = env.evaluate()
                        if video is not None and steps % config.video_every_n_actions == 0:
                            video.append_data(
                                video_frame(env.render_cameras(), config.model.camera_keys)
                            )
                        if final_eval["ever_success"]:
                            break
                    if final_eval["ever_success"]:
                        break
                    obs = env.obs()
                    if config.log_every_chunk:
                        print(
                            f"world={world_index:03d} chunk={chunk:02d} "
                            f"bottles={final_eval['num_bottles_in_bin']}/{final_eval['num_active_bottles']} "
                            f"success={final_eval['ever_success']}",
                            flush=True,
                        )
            finally:
                if video is not None:
                    video.close()

            world = {
                "world_index": world_index,
                "world_seed": seed,
                "success": bool(final_eval["ever_success"]),
                "final_success": bool(final_eval["success"]),
                "reward": float(final_eval["reward"]),
                "steps": steps,
                "wall_s": time.perf_counter() - t0,
                "randomization": env.randomization,
                "final_task_eval": final_eval,
                "video_path": str(video_path) if video_path is not None else None,
            }
            worlds.append(world)
            print(
                f"world={world_index:03d} done success={world['success']} "
                f"bottles={final_eval['max_bottles_in_bin_so_far']}/{final_eval['num_active_bottles']} "
                f"steps={steps}",
                flush=True,
            )
    finally:
        env.close()

    success = np.asarray([w["success"] for w in worlds], dtype=bool)
    rewards = np.asarray([w["reward"] for w in worlds], dtype=np.float32)
    max_bottles = np.asarray(
        [w["final_task_eval"]["max_bottles_in_bin_so_far"] for w in worlds],
        dtype=np.float32,
    )
    summary = {
        "format": "abc_minimal_put_bottles_eval/v1",
        "checkpoint": str(ckpt_path),
        "prompt": config.prompt,
        "config": asdict(config),
        "resolved_device": device,
        "success_rate": float(success.mean()) if success.size else None,
        "num_success": int(success.sum()),
        "num_worlds": len(worlds),
        "mean_reward": float(rewards.mean()) if rewards.size else None,
        "mean_max_bottles_in_bin": float(max_bottles.mean()) if max_bottles.size else None,
        "worlds": worlds,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print(
        f"summary: success_rate={summary['success_rate']} "
        f"num_success={summary['num_success']}/{summary['num_worlds']} "
        f"mean_reward={summary['mean_reward']} "
        f"mean_max_bottles={summary['mean_max_bottles_in_bin']}",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    return summary


def main(config: SimEvalConfig) -> None:
    run_eval(config)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(SimEvalConfig))
