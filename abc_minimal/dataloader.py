"""The dataloader for ABC.
"""

import json
from bisect import bisect_right
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from abc_minimal.config import DiTConfig, PromptConfig, VLAModelConfig
from abc_minimal.dit import task_name_to_prompt
from abc_minimal.operator import operator_label_for
from abc_minimal.preprocess import augment_and_normalize, normalize, preset_for_backbone

# --- data-parallel placement ---------------------------------------------------

@dataclass(frozen=True)
class DataParallelScope:
    rank: int
    world: int
    placement: str
    checkpoint_writer: bool

def read_shard_marker(cache_root: Path):
    marker = cache_root / "hf_status" / "shard.json"
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return None
    return data if data.get("data_placement") == "node_sharded" else None

def cache_is_node_sharded(cache_root: Path) -> bool:
    return read_shard_marker(cache_root) is not None

def check_shard_consistency(marker, world):
    """Every node must have sharded the same dataset snapshot the same way.
    Nodes cannot see each other's disks, so compare markers over the
    process group and fail fast instead of training on overlapping shards."""
    shared_keys = ("repo_id", "revision", "num_nodes", "tasks", "splits",
                   "train_dir", "val_dir", "global_manifest_sha")
    payload = {"shared": {k: marker.get(k) for k in shared_keys},
               "node_rank": marker.get("node_rank")}
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    ref = gathered[0]["shared"]
    problems = []
    mismatched = sorted({g["node_rank"] for g in gathered if g["shared"] != ref})
    if mismatched:
        problems.append(f"nodes {mismatched} disagree with node "
                        f"{gathered[0]['node_rank']} on the shard config {ref}")
    num_nodes = ref.get("num_nodes")
    node_ranks = sorted({g["node_rank"] for g in gathered})
    if isinstance(num_nodes, int) and node_ranks != list(range(num_nodes)):
        problems.append(f"markers say num_nodes={num_nodes} but node ranks "
                        f"present are {node_ranks}")
    if problems:
        raise RuntimeError("inconsistent node-sharded caches: " + "; ".join(problems))

def data_parallel_scope(cache_root: Path, rank, world, local_rank, local_world):
    if cache_is_node_sharded(cache_root):
        return DataParallelScope(
            rank=local_rank,
            world=local_world,
            placement="node_sharded",
            checkpoint_writer=local_rank == 0,
        )
    return DataParallelScope(
        rank=rank,
        world=world,
        placement="shared",
        checkpoint_writer=rank == 0,
    )


# --- on-disk episode format ----------------------------------------------------

def scan_episodes(
    data_dir, default_task_name, model_config: DiTConfig | VLAModelConfig
):
    """Return episode metadata needed for frame sampling and video splitting."""
    episodes = []
    unlabelled = []
    row_width = model_config.state_dim + model_config.action_dim
    for ep_dir in sorted(Path(data_dir).iterdir()):
        bin_path = ep_dir / "states_actions.bin"
        if not bin_path.exists():
            continue
        length = bin_path.stat().st_size // (row_width * 8)
        usable = length - (model_config.chunk_length - 1)
        if usable <= 0:
            continue
        meta = {}
        if (ep_dir / "episode_metadata.json").exists():
            meta = json.loads((ep_dir / "episode_metadata.json").read_text())
        cams = meta.get("cameras") or model_config.camera_keys
        task_name = meta.get("task_name") or default_task_name
        timeline = _prompt_timeline(ep_dir, meta, task_name)
        if timeline is None:
            unlabelled.append(ep_dir.name)
            continue
        episodes.append((ep_dir, length, usable, tuple(cams), task_name, timeline))
    if unlabelled:
        print(f"[prompt] {Path(data_dir).name}: skipped {len(unlabelled)} episode(s) the "
              f"release could not label ({', '.join(unlabelled[:3])})", flush=True)
    return episodes


def _prompt_timeline(ep_dir, meta, task_name):
    """The episode's prompts as ``((frame, prompt), ...)``, each holding from its
    frame until the next.

    Sequence tasks (sim ``multi_drawer_search``) prompt 2-4 targets in order and
    advance mid-episode, so they carry a ``prompt_timeline``; every other episode
    gets the single entry ``_episode_prompt`` derives. There is deliberately no
    whole-episode prompt: collapsing a timeline to one string is what labelled
    every drawer frame with the first target. None means the release could not
    reconstruct this episode's timeline, so the caller drops it.
    """
    timeline = meta.get("prompt_timeline")
    if timeline:
        return tuple((int(e["frame"]), e["prompt"]) for e in timeline)
    if meta.get("prompt_source", {}).get("status") == "excluded":
        return None
    return ((0, _episode_prompt(ep_dir, meta, task_name) or task_name_to_prompt(task_name)),)


def _prompt_at(timeline, frame_idx):
    """The prompt holding at frame ``frame_idx``."""
    frames = [frame for frame, _ in timeline]
    return timeline[max(bisect_right(frames, int(frame_idx)) - 1, 0)][1]


def _episode_prompt(ep_dir, meta, task_name):
    """The episode's directive from episode_metadata's instruction field, or
    None for the task_name derivation.
    """
    instruction = meta.get("instruction")
    if (
        isinstance(instruction, str)
        and instruction.strip()
        and task_name_to_prompt(instruction) != task_name_to_prompt(task_name or "")
    ):
        return instruction.strip()

    recorded = _randomization_prompt(ep_dir)
    if recorded is not None and task_name_to_prompt(recorded) != task_name_to_prompt(
        task_name or ""
    ):
        raise ValueError(
            f"{ep_dir.name}: randomization.json records the directive "
            f"{recorded!r} but episode_metadata.json's instruction does not "
            "carry it, so this cache predates the instruction prompt schema. "
            "Re-download the task's episodes (the published sim_224 tars now "
            "mirror the directive into instruction), or copy randomization's "
            "metadata prompt into each episode_metadata.json instruction field."
        )
    return None


def _randomization_prompt(ep_dir):
    """The prompt randomization.json recorded, used only to validate that a
    cache's episode_metadata carries the directive (see _episode_prompt)."""
    rand_path = ep_dir / "randomization.json"
    if not rand_path.exists():
        return None
    try:
        metadata = json.loads(rand_path.read_text()).get("metadata") or {}
    except (json.JSONDecodeError, OSError):
        return None
    prompt = metadata.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None

def read_state_action_rows(
    ep_dir, start, end, model_config: DiTConfig | VLAModelConfig
):
    row_width = model_config.state_dim + model_config.action_dim
    row_bytes = row_width * 8
    with open(ep_dir / "states_actions.bin", "rb") as f:
        f.seek(start * row_bytes)
        raw = f.read((end - start) * row_bytes)
    return np.frombuffer(raw, dtype=np.float64).reshape(-1, row_width)

_KEYFRAME_CACHE: dict = {}


def _mp4_sync_samples(path):
    """1-based keyframe sample numbers from the mp4's stss box, or None.

    The container records exactly which samples are sync samples; reading it
    is a few kilobytes, where re-probing the whole stream per sample is not.
    An absent stss box means every sample is a keyframe (per ISO 14496-12),
    which the caller treats as "no mapping needed".
    """
    import struct

    def walk(f, start, end, chain):
        f.seek(start)
        while f.tell() < end:
            head = f.tell()
            header = f.read(8)
            if len(header) < 8:
                return None
            size, kind = struct.unpack(">I4s", header)
            header_len = 8
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                header_len = 16
            if size == 0:
                size = end - head
            if size < header_len:  # malformed; let the caller fall back
                return None
            body = head + header_len
            if kind == chain[0]:
                if len(chain) == 1:
                    f.seek(body + 4)  # version/flags
                    (count,) = struct.unpack(">I", f.read(4))
                    return list(struct.unpack(f">{count}I", f.read(4 * count)))
                found = walk(f, body, head + size, chain[1:])
                if found is not None:
                    return found
            f.seek(head + size)
        return None

    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            file_end = f.tell()
            return walk(f, 0, file_end, [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stss"])
    except (OSError, struct.error):
        return None


def decode_frame(ep_dir, idx, episode_length, source_cameras, camera_keys):
    """Decode combined-video frame idx via torchcodec with a synthesized CFR
    frame map (pts = 512*k, 1/15360 timebase) whose keyframe flags come from
    the file's own sync-sample table.

    `source_cameras` is the actual stack order in combined mp4. Stereo episodes
    deterministically alias one top eye to `top`, matching production export.
    """
    import hashlib

    from torchcodec.decoders import VideoDecoder

    video_path = ep_dir / "combined_camera-images-rgb.mp4"
    keyframes = _KEYFRAME_CACHE.get(ep_dir)
    if keyframes is None:
        sync = _mp4_sync_samples(video_path)
        keyframes = frozenset(n - 1 for n in sync) if sync else frozenset()
        _KEYFRAME_CACHE[ep_dir] = keyframes

    if keyframes:
        frames = [
            {"pts": 512 * i, "duration": 512, "key_frame": 1 if i in keyframes else 0}
            for i in range(episode_length)
        ]
        decoder = VideoDecoder(
            str(video_path), custom_frame_mappings=json.dumps({"frames": frames})
        )
    else:
        # No sync-sample table to trust: let the decoder probe the stream.
        decoder = VideoDecoder(str(video_path))
    frame = decoder[idx]  # (C, n_cams * H, W) uint8
    n_cams = len(source_cameras)
    h = frame.shape[1] // n_cams
    cams_out = {
        name: frame[:, i * h : (i + 1) * h, :].float() / 255.0
        for i, name in enumerate(source_cameras)
    }
    if "top" not in cams_out and "top_left" in cams_out and "top_right" in cams_out:
        digest = hashlib.sha1(ep_dir.name.encode("utf-8")).digest()[0]
        cams_out["top"] = cams_out["top_left" if digest % 2 == 0 else "top_right"]
    return {cam: cams_out[cam] for cam in camera_keys}


# --- datasets and sampling -----------------------------------------------------

class EpisodeDataset(Dataset):
    """Map-style dataset over all usable (episode, frame) pairs."""

    def __init__(
        self,
        data_dir,
        norm_stats,
        train,
        default_task_name,
        mask_state_ratio,
        model_config: DiTConfig | VLAModelConfig,
        prompt_config: PromptConfig | None = None,
        operator_label_maps=None,
        norm_preset="auto",
    ):
        self.episodes = scan_episodes(data_dir, default_task_name, model_config)
        if not self.episodes:
            raise ValueError(f"no episodes found in {data_dir}")
        self.model_config = model_config
        self.camera_keys = tuple(model_config.camera_keys)
        self.norm_stats = norm_stats
        self.train = train
        self.mask_state_ratio = mask_state_ratio
        backbone = getattr(model_config, "backbone", None)
        self.image_size = getattr(backbone, "image_size", 224)
        # "auto" derives the image-normalization preset from the vision backbone
        # (DiT path). The VLA path passes norm_preset=None so SigLIP owns image
        # normalization and frames stay in raw [0, 1].
        self.norm_preset = (
            preset_for_backbone(model_config.vision_backbone)
            if norm_preset == "auto"
            else norm_preset
        )
        # Prompt composition rules; dropouts only apply when train=True.
        self.prompt_config = prompt_config or PromptConfig()
        # {task_name: {operator_uuid: rank}} — empty dict = no operator conditioning.
        self.operator_label_maps = operator_label_maps or {}
        # Per-episode sidecar caches, filled lazily as episodes are first sampled.
        self._subtask_cache = {}
        self._operator_cache = {}
        self.cum = np.cumsum([usable for _, _, usable, _, _, _ in self.episodes])

    def __len__(self):
        return int(self.cum[-1])

    def sample(self, rng):
        global_idx = int(rng.integers(0, int(self.cum[-1])))
        return self[global_idx]

    def __getitem__(self, global_idx):
        ep_idx = int(np.searchsorted(self.cum, global_idx, side="right"))
        k = int(global_idx - (self.cum[ep_idx - 1] if ep_idx > 0 else 0))
        ep_dir, length, _, source_cameras, task_name, prompt_timeline = self.episodes[ep_idx]

        rows = read_state_action_rows(
            ep_dir, k, k + self.model_config.chunk_length, self.model_config
        )
        state = normalize(rows[0, : self.model_config.state_dim], self.norm_stats["state"])
        state = state.astype(np.float32)
        actions = normalize(rows[:, self.model_config.state_dim :], self.norm_stats["actions"])
        actions = actions.astype(np.float32)

        state_is_masked = bool(self.train and torch.rand(1).item() < self.mask_state_ratio)
        if state_is_masked:
            state = np.zeros_like(state)

        images = augment_and_normalize(
            decode_frame(ep_dir, k, length, source_cameras, self.camera_keys),
            self.train,
            norm_preset=self.norm_preset,
            image_size=self.image_size,
        )
        prompt = self.build_prompt(ep_dir, k, task_name, prompt_timeline)
        return {
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "images": images,
            "state_is_masked": state_is_masked,
            "prompt": prompt,
        }

    def build_prompt(self, ep_dir, frame_idx, task_name, prompt_timeline):
        """Compose the text prompt for one frame: the episode's prompt at that
        frame (see _prompt_timeline), plus optional subtask and operator labels.
        """
        base = _prompt_at(prompt_timeline, frame_idx)
        prompt = self._apply_subtask(base, ep_dir, frame_idx)
        prompt = self._apply_operator(prompt, ep_dir, task_name)
        return prompt

    def _apply_subtask(self, base, ep_dir, frame_idx):
        if not self.prompt_config.use_subtask_as_prompt:
            return base
        subtask = self._subtask_label(ep_dir, frame_idx)
        if not subtask:
            return base
        if self.train and torch.rand(1).item() < self.prompt_config.subtask_dropout_prob:
            return base
        if self.prompt_config.subtask_mode == "append":
            return self.prompt_config.subtask_append_format.format(prompt=base, subtask=subtask)
        return task_name_to_prompt(subtask)  # "replace" (default)

    def _apply_operator(self, prompt, ep_dir, task_name):
        if not self.prompt_config.use_operator_id_as_prompt or not self.operator_label_maps:
            return prompt
        label = operator_label_for(
            task_name, self._operator_id(ep_dir), self.operator_label_maps,
            mode=self.prompt_config.operator_prompting_mode,
        )
        if not label:
            return prompt
        if self.train and torch.rand(1).item() < self.prompt_config.operator_dropout_prob:
            return prompt
        return self.prompt_config.operator_append_format.format(prompt=prompt, operator=label)

    def _subtask_label(self, ep_dir, frame_idx):
        if ep_dir not in self._subtask_cache:
            path = ep_dir / "subtasks.json"
            self._subtask_cache[ep_dir] = json.loads(path.read_text()) if path.exists() else {}
        return self._subtask_cache[ep_dir].get(str(frame_idx), "")

    def _operator_id(self, ep_dir):
        """Per-episode operator id from operator.json, falling back to
        episode_metadata.json. Cached; empty string when neither carries one."""
        if ep_dir not in self._operator_cache:
            op = ""
            op_path = ep_dir / "operator.json"
            if op_path.exists():
                op = json.loads(op_path.read_text()).get("operator_id", "")
            if not op:
                meta_path = ep_dir / "episode_metadata.json"
                if meta_path.exists():
                    op = json.loads(meta_path.read_text()).get("operator_id", "")
            self._operator_cache[ep_dir] = op or ""
        return self._operator_cache[ep_dir]

class MixtureDataset(Dataset):
    """Train-time mixture: each draw picks a component by `weights`, then a
    uniform usable-frame sample within that component.
    """

    def __init__(self, components, weights, length, seed=0):
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.length = int(length)
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx))
        comp_idx = int(rng.choice(len(self.components), p=self.weights))
        return self.components[comp_idx].sample(rng)

class GlobalStepSampler(Sampler):
    """Map completed training steps to a deterministic per-rank sample stream."""

    def __init__(self, start_step, stop_step, batch_size, data_rank, data_world):
        self.start = int(start_step) * int(batch_size)
        self.count = max(0, (int(stop_step) - int(start_step)) * int(batch_size))
        self.data_rank = int(data_rank)
        self.data_world = int(data_world)

    def __iter__(self):
        for sample_idx in range(self.start, self.start + self.count):
            yield sample_idx * self.data_world + self.data_rank

    def __len__(self):
        return self.count

def collate(samples, camera_keys):
    return {
        "state": torch.stack([s["state"] for s in samples]),
        "actions": torch.stack([s["actions"] for s in samples]),
        "images": {
            cam: torch.stack([s["images"][cam] for s in samples]) for cam in camera_keys
        },
        "state_is_masked": torch.tensor([s["state_is_masked"] for s in samples]),
        "prompt": [s["prompt"] for s in samples],
    }


# --- loader construction -------------------------------------------------------

def build_train_loader(config, components, norm_stats, data_scope, resume_step,
                       model_config, operator_label_maps=None, norm_preset="auto"):
    """Build the train mixture and its step-keyed loader.

    Returns the loader plus the per-component datasets for logging."""
    prompt_config = getattr(config, "prompt", None)
    train_components = [
        EpisodeDataset(Path(config.cache_root) / c.train_dir, norm_stats, train=True,
                       default_task_name=c.task_name,
                       mask_state_ratio=config.flow.mask_state_ratio,
                       model_config=model_config,
                       prompt_config=prompt_config,
                       operator_label_maps=operator_label_maps,
                       norm_preset=norm_preset)
        for c in components
    ]
    component_weights = [c.weight for c in components]
    mixture_length = sum(len(d) for d in train_components)
    train_ds = MixtureDataset(
        train_components, component_weights, mixture_length, seed=config.seed
    )
    train_sampler = GlobalStepSampler(
        start_step=resume_step,
        stop_step=config.train_steps,
        batch_size=config.batch_size,
        data_rank=data_scope.rank,
        data_world=data_scope.world,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate, camera_keys=model_config.camera_keys),
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, train_components

def build_val_loaders(config, components, norm_stats, data_scope, model_config,
                      operator_label_maps=None, norm_preset="auto"):
    """Build per-component val loaders striding this rank's slice.

    Returns the loaders plus (name, dataset) pairs for logging."""
    prompt_config = getattr(config, "prompt", None)
    val_components = [
        (c.val_dir, EpisodeDataset(Path(config.cache_root) / c.val_dir, norm_stats,
                                   train=False,
                                   default_task_name=c.task_name,
                                   mask_state_ratio=config.flow.mask_state_ratio,
                                   model_config=model_config,
                                   prompt_config=prompt_config,
                                   operator_label_maps=operator_label_maps,
                                   norm_preset=norm_preset))
        for c in components
    ]
    val_loaders = {}
    for name, val_ds in val_components:
        # The eval budget has to be spread across the WHOLE split: taking the
        # first budget-many indices evaluates a couple of episodes' consecutive
        # frames (720 frames ≈ 2 episodes at defaults) and silently miscovers
        # val while reporting itself as the split's number.
        budget = config.val_batches * config.batch_size * data_scope.world
        if len(val_ds) <= budget:
            all_indices = range(len(val_ds))
        else:
            step = len(val_ds) / budget
            all_indices = (int(i * step) for i in range(budget))
        val_indices = list(all_indices)[data_scope.rank :: data_scope.world]
        print(
            f"[val] {name}: {len(val_indices)} of {len(val_ds)} frames per rank, "
            f"evenly strided across the split",
            flush=True,
        )
        val_loaders[name] = DataLoader(
            torch.utils.data.Subset(val_ds, val_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=partial(collate, camera_keys=model_config.camera_keys),
            drop_last=True,
        )
    return val_loaders, val_components
