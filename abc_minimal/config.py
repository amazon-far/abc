"""Training and model configuration dataclasses."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class OptimConfig:
    """AdamW with a linear-warmup-then-constant LR schedule."""
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    vision_lr_scale: float = 1.0


@dataclass
class FlowConfig:
    """Rectified-flow matching + action-prefix conditioning."""
    mask_state_ratio: float = 0.1
    max_action_prefix: int = 4
    prefix_conditioning_prob: float = 1.0
    prefix_noise_scale: float = 0.05
    num_diffusion_steps: int = 10


@dataclass
class ClipConfig:
    """CLIP ViT-B/32 text asset locations."""
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".cache" / "clip"))
    model_url: str = (
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    )
    bpe_url: str = (
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    )
    model_name: str = "ViT-B-32.pt"
    bpe_name: str = "bpe_simple_vocab_16e6.txt.gz"


@dataclass
class MixtureComponent:
    """One source in the train/val mixture."""
    train_dir: str
    val_dir: str
    weight: float
    task_name: str


@dataclass
class DiTConfig:
    """ABC-DiT architecture defaults."""
    hidden_size: int = 1536
    depth: int = 32
    num_heads: int = 24
    mlp_ratio: float = 4.0
    state_dim: int = 14
    action_dim: int = 14
    chunk_length: int = 30
    camera_keys: tuple[str, ...] = ("top", "left", "right")
    task_embed_dim: int = 512

    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    vision_pool_num_queries: int = 12
    vision_pool_num_heads: int = 8
    vision_pool_mlp_ratio: int = 4


# Reference hours-weighted real+sim bottles mix.
MIXTURE_PRESETS: dict[str, list[MixtureComponent]] = {
    "bottles": [
        MixtureComponent("train_real", "val_real", 0.8172, "throw_plastic_bottles_in_bin"),
        MixtureComponent("train_sim", "val_sim", 0.1828, "sim_put_the_plastic_bottles_in_the_bin"),
    ],
}


@dataclass
class TrainConfig:
    """Minimal ABC-DiT bottles-in-bin training."""
    cache_root: str = field(
        default_factory=lambda: os.environ.get("ABC_CACHE", "/tmp/abc_minimal_cache")
    )
    seed: int = 123
    batch_size: int = 90
    num_workers: int = 16
    train_steps: int = 75_000

    mixture_preset: Literal["bottles"] = "bottles"
    mixture: list[MixtureComponent] = field(default_factory=list)

    load_pretrained: bool = False
    dino_bf16: bool = True
    compile: bool = True

    log_every: int = 20
    val_every: int = 2500
    val_batches: int = 4
    ckpt_every: int = 5000
    log_wandb: bool = False
    wandb_project: str = "minimal-abc"

    optim: OptimConfig = field(default_factory=OptimConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)

    def resolve_mixture(self) -> list[MixtureComponent]:
        return self.mixture if self.mixture else MIXTURE_PRESETS[self.mixture_preset]


@dataclass
class SimEvalConfig:
    """MuJoCo-Warp put-bottles evaluation."""
    checkpoint: str
    norm_stats_path: str | None = None
    output_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[1] / "outputs" / "sim_eval_put_bottles"
        )
    )
    num_worlds: int = 5
    seed: int = 20260511
    num_chunks: int = 60
    execute_chunk_dim: int = 15
    diffusion_steps: int = 10
    policy_seed: int = 0
    camera_height: int = 168
    camera_width: int = 224
    device: str = "auto"
    gpu_id: int | None = None
    log_every_chunk: bool = False
    save_video: bool = False
    video_fps: int = 30
    video_every_n_actions: int = 1
    prompt: str = "sim put the plastic bottles in the bin"

    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)


@dataclass
class VizPolicyConfig:
    """Live viser viewer over a single ABC-DiT sim rollout."""
    sim: SimEvalConfig
    port: int = 8080
    fast_inference: bool = True
    fast_compile_mode: str = "max-autotune-no-cudagraphs"


def validate_model_config(model: DiTConfig) -> list[str]:
    model_dims = [
        model.hidden_size, model.depth, model.num_heads, model.mlp_ratio,
        model.state_dim, model.action_dim, model.chunk_length, model.task_embed_dim,
        model.vit_embed_dim, model.vit_depth, model.vit_num_heads,
        model.vision_pool_num_queries, model.vision_pool_num_heads,
        model.vision_pool_mlp_ratio,
    ]
    errors = []
    if min(model_dims) <= 0 or not model.camera_keys:
        errors.append("model dimensions and camera_keys must be positive/non-empty")
    if (
        model.hidden_size % model.num_heads
        or model.vit_embed_dim % model.vit_num_heads
        or model.vit_embed_dim % model.vision_pool_num_heads
        or model.hidden_size % 2
        or (model.vit_embed_dim // model.vit_num_heads) % 4
    ):
        errors.append("attention dimensions must be compatible with their head counts")
    return errors


def validate_train_config(
    config: TrainConfig, cache_root: Path, checkpoint_path: Path
) -> list[MixtureComponent]:
    components = config.resolve_mixture()
    weights = [c.weight for c in components]
    errors = []

    if (
        min(config.batch_size, config.train_steps, config.log_every, config.val_every,
            config.val_batches, config.ckpt_every) <= 0
        or config.num_workers < 0
    ):
        errors.append(
            "batch size, step intervals, and val_batches must be positive; "
            "num_workers must be non-negative"
        )
    if (
        not 0 <= config.flow.mask_state_ratio <= 1
        or not 0 <= config.flow.prefix_conditioning_prob <= 1
        or config.flow.prefix_noise_scale < 0
    ):
        errors.append(
            "flow probabilities must be in [0, 1] and prefix_noise_scale must be non-negative"
        )
    errors.extend(validate_model_config(config.model))
    if (
        not components
        or any(not math.isfinite(w) or w <= 0 for w in weights)
        or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        errors.append(
            f"mixture weights must be positive and sum to 1.0, "
            f"got {sum(weights) if weights else 0:.8g}"
        )

    required = [cache_root / "norm_stats.json"]
    required += [cache_root / p for c in components for p in (c.train_dir, c.val_dir)]
    if config.load_pretrained:
        required.append(checkpoint_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        errors.append("missing required paths: " + ", ".join(missing))

    if errors:
        raise ValueError("Invalid training config:\n  - " + "\n  - ".join(errors))
    return components
