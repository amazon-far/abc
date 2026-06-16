# **Scalable Behavior Cloning with Open Data, Training, and Evaluation**

<p align="center">
  <strong>
    <a href="https://abc.bot">Project Website</a> |
    <a href="todo">Paper</a> |
    <a href="todo">Raw Data</a>
  </strong>
</p>

![](assets/teaser.jpg)


Code for the ABC project. Currently this codebase allows you to train a single-task ABC-DiT policy which can deploy in real and in sim on the put bottles in bin task. 

## Release Roadmap

> Note: we have released a minimal training pipeline for ABC-DiT & conversion scripts for the data. We also re-host a small subset of the sim data and real data for 1 task to allow users to get started quickly. Please check back later for the full code release, including VLA training, real deployment infra & pretrained checkpoints.

- [x] June 17 -- Release Minimal Training Pipeline
- [ ] End of June -- Release all sim data 
- [ ] By end of July -- full code release

## Setup

```bash
# Install uv if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Install ffmpeg.
sudo apt-get install -y ffmpeg     # on Linux
```

```bash
# Pin Python and create the project venv. uv reads pyproject.toml here.
cd abc
uv python pin 3.12
uv sync
```

## Training

First we need to download the requisite data (norm stats and either a sample or full data.)
```bash
uv run prepare.py # to download preview (a few episodes of data, ~130Mb)
uv run prepare.py --full # to download all data for bottles in bin (~35Gb)
```

This populates the cache dir (default `/tmp/abc_minimal_cache`) with:

```
$ABC_CACHE/
  norm_stats.json                       # state/action z-score stats
  train_real/episode_<uuid>/{states_actions.bin, combined_camera-images-rgb.mp4, episode_metadata.json}
  val_real/...
  train_sim/...
  val_sim/...
```

:warning: Note: `prepare.py` does not download DINO weights. Review and follow the DINO license terms, then download the weights from [Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) or [Hugging Face](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m). Save the file as `dinov3_vitb16_pretrain_lvd1689m.pth` in the cache dir. :warning:

The command to run training is below. Note that this is for single node training with 8 GPUs, change `nproc-per-node` if you want.

```bash
uv run torchrun --standalone --nproc-per-node 8 train.py
```
The dataclass config is exposed as CLI flags; `uv run python train.py --help`
shows training, optimizer, flow, CLIP asset, and model options such as
`--model.hidden-size`, `--model.depth`, and `--model.camera-keys`. The default
model config is the checkpoint-compatible ABC-DiT XL shape.

Training defaults in `abc_minimal/config.py` match the production reference
finetune (lr 1e-4 with a 1k-step linear warmup, AdamW(0.9, 0.95), wd 0.01,
grad clip 10, prefix conditioning max 4 with noise 0.05, 10% state masking,
batch 90/GPU, 75k steps, hours-weighted 2-component mixture).
The dataclass config is exposed as CLI flags; `uv run python train.py --help`
shows training, optimizer, flow, CLIP asset, and model options such as
`--model.hidden-size`, `--model.depth`, and `--model.camera-keys`. The default
model config is the checkpoint-compatible ABC-DiT XL shape.

## Evaluation

Te visualize your trained policy:

```bash
uv run viz_policy.py --sim.checkpoint $ABC_CACHE/finetune_checkpoints/last.pt \
                     --port 8080
```

opens a viser window at `localhost:8080` which you can view your policy in.

`eval_sim.py` runs a more systematic evaluation.

```bash
# 20 worlds, save a video of each rollout, log per-chunk progress.
uv run eval_policy.py \
    --checkpoint $ABC_CACHE/finetune_checkpoints/last.pt \
    --num-worlds 20 \
    --save-video --log-every-chunk
```


```bash
# Output: $REPO/outputs/sim_eval_put_bottles/
#   summary.json     — success_rate, num_success, mean_max_bottles_in_bin
#   world_*.mp4      — per-world rollout videos (with --save-video)
```

Useful flags:

- `--num-worlds N` — independent random scenes (default 5).
- `--num-chunks N` — action chunks per rollout; each chunk is
`--execute-chunk-dim` actions (defaults: 60 chunks × 15 = 900 sim steps).
- `--diffusion-steps N` — flow-matching Euler steps per inference
(default 10, matches production).
- `--checkpoint` — accepts a local `.pt` path or `s3://…/<file>.pt`.
- `--norm-stats-path` — explicit `norm_stats.json` (otherwise uses the
one bundled in the checkpoint).

First launch compiles MJWarp's CUDA kernels (~1 min on H100; cached for
subsequent worlds within the same process).

## Episode exports & training data format

:warning: *TODO (arthur)* -- document how to download a task data etc from HF once it's up on xdof side

To convert release-format MCAP episodes into the training format, point
`export_mcap.py` at a root directory containing task folders:

```bash
uv run export_mcap.py ./train_run_1 ./out
```

The input is expected to look like:

```text
train_run_1/
  <task_name>/
    episode_<uuid>/
      episode.mcap
```

You can also pass the number of worker processes:

```bash
uv run export_mcap.py ./train_run_1 ./out 8
```

Each output episode is written to `./out/episode_<uuid>/` in the same format
the trainer reads:

```text
episode_<uuid>/
  states_actions.bin               # (num_steps, 28) float64: 14 states + 14 actions
  combined_camera-images-rgb.mp4   # 30 fps vertical stack of 224x224 camera views
  episode_metadata.json            # task name, cameras, resolutions, timing, num_steps
```

## Licenses

This repository includes and adapts code from the following third-party
projects. Original license files and copyright headers are retained in all
cases. Bundled license texts live under `abc_minimal/third_party/`.

| Project | License | License file | Inclusion | What we use/adapt |
| --- | --- | --- | --- | --- |
| [DINOv3](https://github.com/facebookresearch/dinov3) | DINOv3 License (Meta) | [`abc_minimal/third_party/dinov3/LICENSE.md`](abc_minimal/third_party/dinov3/LICENSE.md) | Adapted (`abc_minimal/dit.py`); pretrained weights downloaded by the user | ViT-B/16 vision backbone (`DinoRope`, `DinoAttention`, `DinoMlp`, etc.) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | MIT | [`abc_minimal/third_party/clip/LICENSE`](abc_minimal/third_party/clip/LICENSE) | Adapted (`abc_minimal/dit.py`); ViT-B/32 text weights + BPE vocab downloaded at runtime | CLIP text encoder + BPE tokenizer (`CLIPBPETokenizer`, `CLIPTextTower`, `CLIPTextEmbedder`) |
| [i2rt YAM](https://github.com/i2rt-robotics) | MIT | [`assets/put_bottles/assets/i2rt_yam/LICENSE`](assets/put_bottles/assets/i2rt_yam/LICENSE) | Vendored under `assets/put_bottles/assets/i2rt_yam/` | YAM robot MuJoCo model, meshes, and scene assets |

### DINOv3 use restrictions

The DINOv3 License prohibits use of the DINO Materials (including weights and
derivatives) for: military purposes; activities subject to ITAR or other
export-control regimes covering defense articles; nuclear applications;
espionage; and the development, manufacture, or use of weapons. Downstream
users who load DINOv3 weights through this codebase are bound by these
restrictions; see `abc_minimal/third_party/dinov3/LICENSE.md` for the full
license text.