"""Train ABC-DiT on the bottles-in-bin dataset.

Builds train/val datasets, runs optimization, validation, and checkpointing.
"""

import json
import math
import os
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from abc_minimal.config import DiTConfig, TrainConfig, validate_train_config
from abc_minimal.dit import (
    CLIPTextEmbedder,
    DiTPolicy,
    load_pretrained,
    task_name_to_prompt,
)
from abc_minimal.preprocess import (
    imagenet_normalize,
    load_norm_stats,
    normalize,
    resize_with_pad,
)

# Enable TF32-backed fp32 matmul on NVIDIA GPUs.
torch.set_float32_matmul_precision("high")


def scan_episodes(data_dir, default_task_name, model_config: DiTConfig):
    """Return episode metadata needed for frame sampling and video splitting."""
    episodes = []
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
        episodes.append((ep_dir, length, usable, tuple(cams), task_name))
    return episodes


def read_state_action_rows(ep_dir, start, end, model_config: DiTConfig):
    row_width = model_config.state_dim + model_config.action_dim
    row_bytes = row_width * 8
    with open(ep_dir / "states_actions.bin", "rb") as f:
        f.seek(start * row_bytes)
        raw = f.read((end - start) * row_bytes)
    return np.frombuffer(raw, dtype=np.float64).reshape(-1, row_width)


def decode_frame(ep_dir, idx, episode_length, source_cameras, camera_keys):
    """Decode combined-video frame idx via torchcodec with a synthesized CFR
    frame map (pts = 512*k, 1/15360 timebase), without per-file probing.

    `source_cameras` is the actual stack order in combined mp4. Stereo episodes
    deterministically alias one top eye to `top`, matching production export.
    """
    import hashlib
    from torchcodec.decoders import VideoDecoder

    frames = [
        {"pts": 512 * i, "duration": 512, "key_frame": 1 if i % 30 == 0 else 0}
        for i in range(episode_length)
    ]
    mapping = json.dumps({"frames": frames})
    decoder = VideoDecoder(
        str(ep_dir / "combined_camera-images-rgb.mp4"), custom_frame_mappings=mapping
    )
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


def _rotate(img_hwc, angle_deg):
    """Rotate (H,W,C) with reflection padding."""
    if abs(angle_deg) < 0.1:
        return img_hwc
    H, W, C = img_hwc.shape
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    grid = torch.stack([gx * cos_a - gy * sin_a, gx * sin_a + gy * cos_a], dim=-1)
    out = F.grid_sample(
        img_hwc.permute(2, 0, 1).unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0)


def augment_and_normalize(images, train):
    """Apply production image augmentations and ImageNet normalization."""
    out = {}
    for cam, img in images.items():
        x = img.permute(1, 2, 0)
        if train and "top" in cam:
            angle = (torch.rand(1) * 10 - 5).item()
            x = _rotate(x, angle)
            H, W, _ = x.shape
            ch, cw = int(H * 0.95), int(W * 0.95)
            if H - ch > 0 and W - cw > 0:
                sh = torch.randint(0, H - ch + 1, (1,)).item()
                sw = torch.randint(0, W - cw + 1, (1,)).item()
                x = x[sh : sh + ch, sw : sw + cw, :]
                x = F.interpolate(
                    x.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
        x = resize_with_pad(x, 224, 224)
        if train:
            b = 0.7 + torch.rand(1).item() * 0.6
            x = x * b
            c = 0.6 + torch.rand(1).item() * 0.8
            mean = x.mean()
            x = (x - mean) * c + mean
            s = 0.5 + torch.rand(1).item() * 1.0
            gray = x.mean(dim=-1, keepdim=True)
            x = gray + (x - gray) * s
            x = torch.clamp(x, 0, 1)
        out[cam] = imagenet_normalize(x.permute(2, 0, 1))
    return out


class EpisodeDataset(Dataset):
    """Map-style dataset over all usable (episode, frame) pairs."""

    def __init__(
        self,
        data_dir,
        norm_stats,
        train,
        default_task_name,
        mask_state_ratio,
        model_config: DiTConfig,
    ):
        self.episodes = scan_episodes(data_dir, default_task_name, model_config)
        if not self.episodes:
            raise ValueError(f"no episodes found in {data_dir}")
        self.model_config = model_config
        self.camera_keys = tuple(model_config.camera_keys)
        self.norm_stats = norm_stats
        self.train = train
        self.mask_state_ratio = mask_state_ratio
        self.cum = np.cumsum([usable for _, _, usable, _, _ in self.episodes])

    def __len__(self):
        return int(self.cum[-1])

    def sample(self, rng):
        global_idx = int(rng.integers(0, int(self.cum[-1])))
        return self[global_idx]

    def __getitem__(self, global_idx):
        ep_idx = int(np.searchsorted(self.cum, global_idx, side="right"))
        k = int(global_idx - (self.cum[ep_idx - 1] if ep_idx > 0 else 0))
        ep_dir, length, _, source_cameras, task_name = self.episodes[ep_idx]

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
            decode_frame(ep_dir, k, length, source_cameras, self.camera_keys), self.train
        )
        return {
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "images": images,
            "state_is_masked": state_is_masked,
            "prompt": task_name_to_prompt(task_name),
        }


class MixtureDataset(Dataset):
    """Train-time mixture: each draw picks a component by `weights`, then a
    uniform usable-frame sample within that component.
    """

    def __init__(self, components, weights, length):
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        comp_idx = int(rng.choice(len(self.components), p=self.weights))
        return self.components[comp_idx].sample(rng)


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


def batch_to_device(batch, device, embedder):
    out = {
        "state": batch["state"].to(device, non_blocking=True),
        "actions": batch["actions"].to(device, non_blocking=True),
        "images": {
            cam: v.to(device, non_blocking=True) for cam, v in batch["images"].items()
        },
        "state_is_masked": batch["state_is_masked"].to(device, non_blocking=True),
        "task_vec_clip": embedder.encode(batch["prompt"]).to(device, non_blocking=True),
    }
    return out


def main(config: TrainConfig):
    cache_root = Path(config.cache_root)
    checkpoint_path = cache_root / "abc_dit_xl_200k_model.pt"
    output_dir = cache_root / "finetune_checkpoints"
    components = validate_train_config(config, cache_root, checkpoint_path)

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    model = DiTPolicy(config.model)
    if config.load_pretrained:
        load_pretrained(model, checkpoint_path)
    else:
        dinov3_ckpt = cache_root / "dinov3_vitb16_pretrain_lvd1689m.pth"
        if dinov3_ckpt.exists():
            sd = torch.load(dinov3_ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            target = model.img_backbone.dinov3_model
            missing, unexpected = target.load_state_dict(sd, strict=False)
            if rank == 0:
                print(f"loaded DINOv3 from {dinov3_ckpt} "
                      f"(missing={len(missing)} unexpected={len(unexpected)})")
        elif rank == 0:
            print(f"no {dinov3_ckpt}, using random DINOv3")
    if config.dino_bf16:
        model.img_backbone.set_bfloat16(True)
        if rank == 0:
            print("DINOv3 bf16 autocast enabled")

    model = model.to(device)

    vision_params = list(model.img_backbone.parameters())
    vision_ids = {id(p) for p in vision_params}
    main_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in vision_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": config.optim.learning_rate},
            {"params": vision_params,
             "lr": config.optim.learning_rate * config.optim.vision_lr_scale},
        ],
        betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        eps=config.optim.adam_epsilon,
        weight_decay=config.optim.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / config.optim.lr_warmup_steps, 1.0)
    )

    if config.compile:
        model = torch.compile(model, fullgraph=True)
        if rank == 0:
            print("torch.compile(fullgraph=True) enabled")

    if distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            bucket_cap_mb=256,
        )
    module = model.module if distributed else model
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod

    norm_stats = load_norm_stats(cache_root / "norm_stats.json")
    if distributed and rank != 0:
        dist.barrier()
    embedder = CLIPTextEmbedder(config.clip, device="cpu")
    if distributed and rank == 0:
        dist.barrier()

    train_components = [
        EpisodeDataset(cache_root / c.train_dir, norm_stats, train=True,
                       default_task_name=c.task_name,
                       mask_state_ratio=config.flow.mask_state_ratio,
                       model_config=config.model)
        for c in components
    ]
    component_weights = [c.weight for c in components]
    mixture_length = sum(len(d) for d in train_components)
    train_ds = MixtureDataset(train_components, component_weights, mixture_length)

    val_components = [
        (c.val_dir, EpisodeDataset(cache_root / c.val_dir, norm_stats, train=False,
                                   default_task_name=c.task_name,
                                   mask_state_ratio=config.flow.mask_state_ratio,
                                   model_config=config.model))
        for c in components
    ]

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=config.seed, drop_last=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        collate_fn=partial(collate, camera_keys=config.model.camera_keys),
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loaders = {}
    for name, val_ds in val_components:
        val_indices = list(
            range(rank, min(len(val_ds), config.val_batches * config.batch_size * world), world)
        )
        val_loaders[name] = DataLoader(
            torch.utils.data.Subset(val_ds, val_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=partial(collate, camera_keys=config.model.camera_keys),
            drop_last=True,
        )

    wandb = None
    if config.log_wandb and rank == 0:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(project=config.wandb_project, config=asdict(config))
        except Exception as e:
            print(f"wandb disabled: {e}")

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for c, ds in zip(components, train_components):
            prompts = sorted({task_name_to_prompt(t) for *_, t in ds.episodes})
            print(f"train[{c.train_dir}] weight={c.weight:.4f}: "
                  f"{len(ds.episodes)} episodes, {len(ds)} usable frames, prompts={prompts}")
        for name, ds in val_components:
            print(f"val[{name}]: {len(ds.episodes)} episodes")
        print(f"world={world}")

    model.train()
    step, epoch = 0, 0
    t_last = time.monotonic()
    while step < config.train_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= config.train_steps:
                break
            batch = batch_to_device(batch, device, embedder)

            loss = model(
                batch,
                max_action_prefix=config.flow.max_action_prefix,
                prefix_conditioning_prob=config.flow.prefix_conditioning_prob,
                prefix_noise_scale=config.flow.prefix_noise_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.log_every == 0:
                loss_d = loss.detach()
                if distributed:
                    dist.all_reduce(loss_d, op=dist.ReduceOp.AVG)
                if rank == 0:
                    dt = time.monotonic() - t_last
                    t_last = time.monotonic()
                    sps = config.log_every / dt
                    lr = scheduler.get_last_lr()[0]
                    print(f"step {step:6d}  loss {loss_d.item():.4f}  "
                          f"lr {lr:.2e}  gnorm {grad_norm:.3f}  {sps:.2f} it/s")
                    if wandb:
                        wandb.log({"loss": loss_d.item(), "lr": lr,
                                   "grad_norm": grad_norm.item(),
                                   "steps_per_s": sps}, step=step)

            if step % config.val_every == 0:
                model.eval()
                per_component_recon = {}
                skipped_val = []
                for name, vl in val_loaders.items():
                    rsum, n = 0.0, 0
                    for vb in vl:
                        vb = batch_to_device(vb, device, embedder)
                        with torch.no_grad():
                            pred = module.sample_actions(vb, num_steps=config.flow.num_diffusion_steps)
                            rsum += F.mse_loss(pred, vb["actions"]).item()
                            n += 1
                    stats = torch.tensor([rsum, float(n)], device=device)
                    if distributed:
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    if stats[1].item() > 0:
                        per_component_recon[name] = (stats[0] / stats[1]).item()
                    else:
                        skipped_val.append(name)
                model.train()
                if rank == 0:
                    if per_component_recon:
                        parts = "  ".join(f"{n}={v:.4f}" for n, v in per_component_recon.items())
                        avg = sum(per_component_recon.values()) / len(per_component_recon)
                        print(f"step {step:6d}  val_recon_error {avg:.4f}  ({parts})")
                        if wandb:
                            log = {"val_recon_error": avg}
                            log.update({f"val_recon_error/{n}": v for n, v in per_component_recon.items()})
                            wandb.log(log, step=step)
                    else:
                        print(f"step {step:6d}  val skipped (no full validation batches)")
                    if skipped_val:
                        print(f"step {step:6d}  val skipped components: {', '.join(skipped_val)}")
                t_last = time.monotonic()

            if step % config.ckpt_every == 0 and rank == 0:
                path = output_dir / f"{step}.pt"
                torch.save(
                    {"model": module.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "step": step,
                     "norm_stats": norm_stats},
                    path,
                )
                torch.save(
                    {"model": module.state_dict(), "step": step, "norm_stats": norm_stats},
                    output_dir / "last.pt",
                )
                print(f"saved {path}")
        epoch += 1

    if distributed:
        dist.destroy_process_group()
