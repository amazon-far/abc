# ABC Sim

`abc_sim` is the self-contained simulator package for ABC tasks. It owns the
MuJoCo scenes, task registry, randomization, reward/success evaluation, camera
rendering, and Gym-style environment API. Sim eval, RL, rendered observations,
task randomization, and new sim environments all work from this package alone.

**Contents**

- [Quickstart](#quickstart)
- [Environment API](#environment-api)
- [Step vs Step Chunk](#step-vs-step-chunk)
- [Tasks and Randomization](#tasks-and-randomization)
- [Rewards and Success](#rewards-and-success)
- [Sim Eval](#sim-eval)
  - [Eval flags](#eval-flags)
  - [Batched eval](#batched-eval)
  - [Scored tasks](#scored-tasks)
  - [The put-bottles task and its aliases](#the-put-bottles-task-and-its-aliases)
  - [Prompt defaults](#prompt-defaults)
  - [The multi-task sim policy](#the-multi-task-sim-policy)
  - [Finetuned per-task checkpoints](#finetuned-per-task-checkpoints)
  - [Matching the production eval numbers](#matching-the-production-eval-numbers)
- [Assets](#assets)
- [Rendering](#rendering)
- [Adding a New Env or Task](#adding-a-new-env-or-task)
- [RL Usage](#rl-usage)

## Quickstart

Install project dependencies from the repository root:

```bash
uv sync
```

Download simulator assets before creating environments — either all of them, or
just what one task needs:

```bash
uv run prepare.py --sim
uv run prepare.py --sim-task put_plastic_bottles_in_bin
```

Create and step an environment:

```python
import abc_sim

spec = abc_sim.get_task_spec("put_bottles")
env = abc_sim.make_env(
    task=spec.env_task,
    prompt=spec.prompt,
    camera_height=168,
    camera_width=224,
    max_episode_steps=236 * 15,
    terminate_on_success=True,
)

obs, reset_info = env.reset(seed=0, randomize=True)
obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
env.close()
```

For a runnable example:

```bash
uv run python abc_sim/example_api_rollout.py --task put_bottles --render-cameras
```

## Environment API

`abc_sim.make_env(...)` returns a Gymnasium-compatible MuJoCo environment.

```python
env = abc_sim.make_env(
    task="put_bottles",
    render_cameras=True,
    camera_height=168,
    camera_width=224,
    max_episode_steps=3000,
    terminate_on_success=True,
)
```

The stable API contract is:

- `obs, info = env.reset(seed=..., randomize=True)`
- `obs, reward, terminated, truncated, info = env.step(action_14d)`
- `obs, history, reward, terminated, truncated, info = env.step_chunk(action_chunk)`
- `env.action_space.shape == (14,)`
- `env.chunk_action_space.shape == (chunk_dim, 14)`

The 14D policy-space action/state layout is:

```text
[left_j1..left_j6, left_gripper, right_j1..right_j6, right_gripper]
```

Observations are a dictionary with:

- `state`: `(14,)` float32 policy-space state.
- `images`: per-camera `(3, H, W)` uint8 RGB images when rendering is enabled.
- `prompt`: task prompt string.
- `masks`: per-camera availability masks.
- `camera_timestamps`: per-camera timestamps.

`env.observation_space`, `env.action_space`, and `env.chunk_action_space` are
defined for downstream Gym/RL integrations.

## Step vs Step Chunk

Use `step(...)` when the controller or RL policy emits one action at a time:

```python
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

Use `step_chunk(...)` when a behavior cloning policy emits a fixed action
chunk:

```python
chunk = policy(obs)  # shape: (chunk_dim, 14)
obs, history, reward, terminated, truncated, info = env.step_chunk(chunk)
```

`step_chunk(...)` repeatedly calls the single-action stepping path and returns
the final observation plus a `history` dictionary for the intermediate states.

## Tasks and Randomization

Task metadata lives in `abc_sim/task_specs.py`. Use the registry helpers to
resolve canonical task names, prompts, scene XMLs, randomizers, and evaluators:

```python
spec = abc_sim.get_task_spec("put_bottles")
print(spec.name, spec.env_task, spec.prompt)
```

Randomization is enabled by default for tasks with registered randomizers:

```python
obs, info = env.reset(seed=123, randomize=True)
randomization = info["randomization"]
```

`randomization` contains the sampled world metadata needed for replay or fixed
eval sets. Passing `randomize=False` resets the scene without sampling a new
world.

## Rewards and Success

Task evaluators live under `abc_sim/task_eval/`. If a task has an evaluator,
`env.step(...)` and `env.step_chunk(...)` return task metrics in `info`:

- `info["task_reward"]`: scalar reward.
- `info["task_success"]`: boolean success flag.
- `info["task_eval"]`: evaluator-specific metrics.

Episode stopping is controlled by environment configuration:

- `max_episode_steps`: returns `truncated=True` when the action horizon is hit.
- `terminate_on_success=True`: returns `terminated=True` when the evaluator
  reports success.

You can also query the evaluator directly:

```python
result = env.evaluate_task()
if result is not None:
    print(result.to_info(squeeze=True))
```

## Sim Eval

Policy evaluation drives this same public environment API through the repo's
eval entrypoint, `eval_policy.py`. The default `--task
put_plastic_bottles_in_bin` and every other simulation task run through this
package's catalogue. The old spelling `put_bottles` remains an alias for the
same catalogue task; it no longer selects a separate scene or scoring rule. The
sim assets are not checked in, so download them first (see [Assets](#assets)):

```bash
uv run prepare.py --sim
```

```bash
# Put bottles in the bin, 20 worlds, one video per rollout.
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --task put_plastic_bottles_in_bin \
    --num-worlds 20 --save-video --video-every-n-actions 15

# Any other task in the catalogue, by name or alias.
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --task turn_mug_right_side_up \
    --num-worlds 20 --save-video --video-every-n-actions 15

# Output: $REPO/outputs/sim_eval_<task>/
#   summary.json     — success_rate, num_success (mean_max_bottles_in_bin
#                      only for the bottles tasks, which count bottles)
#   world_*.mp4      — per-world rollout videos (with --save-video)
```

The checkpoints to evaluate — your own finetunes, the 75k-step bottles-only
policy, or the multi-task sim policy — are described in the
[top-level README](../README.md#evaluation).

### Eval flags

- `--num-worlds N` — independent random scenes (default 5; use 50+ for
numbers you want to quote — at a true 10% success rate a 5-world run most
often reads 0).
- `--num-chunks N` — action chunks per rollout; each chunk is
`--execute-chunk-dim` actions (defaults: 236 chunks × 15 = 3540 sim steps,
the horizon the production sim-eval dashboard measured checkpoints with).
- `--rtc` / `--no-rtc` (default on) — RTC evaluation: each inference receives
the next `--rtc-prefix-length` (default 4) not-yet-executed actions and overlaps
inference with their execution. `--no-rtc` runs a synchronous loop.
- `--parallel-worlds N` — step N worlds together in MJWarp physics (default
0: one CPU MuJoCo world at a time). `--num-worlds` must be a multiple of N;
see [Batched eval](#batched-eval).
- `--randomization JSON` — reset request for the task randomizer, applied to
every world, e.g. `'{"bottle_count": 6, "randomize_variants": false,
"randomize_scales": false}'` (fields: `abc_sim/randomization/requests.py`).
- `--diffusion-steps N` — flow-matching Euler steps per inference
(default 10, matches production).
- `--checkpoint` — path to the `.pt` checkpoint to evaluate.
- `--norm-stats-path` — explicit `norm_stats.json` (otherwise uses the
one bundled in the checkpoint).
- `--fast-inference` / `--no-fast-inference` (default on) — bf16 +
torch.compile + CUDA-graph captured `sample_actions`. ~5× faster
inference; first call pays a one-time ~25 s compile cost.
- `--vanilla-physics` / `--no-vanilla-physics` (default off) — no-op:
every task already steps physics in CPU MuJoCo. Kept for command-line
compatibility.

Note that the first launch compiles MJWarp's CUDA kernels (~1 min).

`--camera-backend mujoco` renders with CPU MuJoCo instead of MJWarp (`blender`
renders with Cycles, see [Rendering](#rendering)): slow, but
it runs anywhere MuJoCo does, which makes it the way to smoke-test on a laptop
(including macOS). On a headless Linux box, set `MUJOCO_GL=egl`.

### Batched eval

`--parallel-worlds N` steps N worlds together in MJWarp physics with one
batched policy call per chunk; RTC, prefix conditioning, `--save-video`, and
`summary.json` are unchanged. A batch shares one compiled model, so pin the
randomization that would recompile it (object counts, variants, scales) with
`--randomization`; poses and colors still vary per world. The sorting tasks
batch with `{"park_inactive": true, "bin_visual_style": "stackable"}`, which
keeps every candidate object in the scene and parks the unused ones behind
the robot. Tasks with a per-step runtime (conveyor pick, ball tray, multi-drawer search) and evaluators
that need MjData or per-episode targets are sequential-only.

MJWarp is a different physics engine: absolute success rates differ from the
CPU MuJoCo loop (put-bottles with the 200k policy: 0.71–0.77 batched vs 0.54
sequential on the same pinned scenes), runs are not seed-reproducible, and every
world starts from the home pose. Compare checkpoints within one backend.

On one H100 a 100-world put-bottles eval takes about 8 minutes after a one-off
compile per batch width (~15 min cold, ~1 min cached), against about an hour
sequentially; the ray-traced render (~7 ms per world per chunk) is the ceiling.

### Scored tasks

All but four tasks ship with a success evaluator, so their `success_rate` is a
real measurement. The unscored four — `put_markers_in_top_drawer`,
`put_markers_in_middle_drawer`, `put_markers_in_bottom_drawer`, and
`build_wood_block_tower` — still roll out and record videos, but have no way to
score themselves: they warn at startup and report reward 0 and success 0
throughout. Watch the videos to judge those.

One task is scored differently: `ball_tray_balancing` is a maintenance task
(success means the ball has not been dropped yet), so its rollouts run the full
chunk budget and its `success_rate` is only comparable between runs with equal
`--num-chunks` — a 10-chunk run asks the policy to hold the ball for 5 seconds,
a 30-chunk run for 15.

Task names, aliases, and prompts are listed in `abc_sim/task_specs.py`, and an
unrecognized `--task` fails immediately with the full catalogue in the error.

### The put-bottles task and its aliases

`put_bottles` and `water_bottles` are aliases of the catalogue task
`put_plastic_bottles_in_bin`; every spelling builds the same environment and
scores with the same shared evaluator, which scores each bottle's centre of
mass against the tapered interior measured from the bin's collision mesh, with
a height cap based on the bottle's own collision reach. (Until August 2026,
`put_bottles` selected a separate scene that `eval_policy.py` built itself —
6 bottles near unit scale from a synthetic home pose. That native path is
retired; numbers measured in it live in the archived eval summaries and are
reproducible from git history, and are not comparable with catalogue-task
numbers, which randomize 2–6 bottles, bin scale up to 1.4×, and start from the
real robot's teleoperation pose.)

### Prompt defaults

The prompt defaults to `"sim "` plus the task's own prompt, so
`--task turn_mug_right_side_up` is prompted `sim turn mug right side up`. That is
how sim episodes are labelled in the training mixture. If `model.pt` has a
neighbouring `model.json` metadata sidecar, its `sim_prompt_map` supplies the
exact prompt that checkpoint saw for each task during training. An explicit
`--prompt` wins over the sidecar, and the sidecar wins over the catalogue
default.

### The multi-task sim policy

`abc_dit_xl_200k_model.pt` is the DiT-XL parent behind `--load-pretrained`
finetuning (see the [top-level README](../README.md#abc-dit-training)), trained on the
3.5k-hour real mixture plus 98 hours of sim across five tasks. It is also the
checkpoint to evaluate those five tasks with:

```bash
uv run prepare.py --pretrained --sim
uv run eval_policy.py \
    --checkpoint cache/abc_dit_xl_200k_model.pt \
    --task put_plastic_bottles_in_bin \
    --num-worlds 20 --save-video
```

The five tasks it saw, and the prompt each one's episodes carried:

| `--task` | prompt it trained under |
| --- | --- |
| `put_plastic_bottles_in_bin` | `sim throw plastic bottles in bin` |
| `sweep_away_paper_scraps_from_table` | `sim sweep away paper scraps from the table` |
| `turn_mug_right_side_up` | `sim turn mug right side up` |
| `load_plates_into_dish_rack` | `sim load plates into tabletop dish rack` |
| `hang_mug_on_mug_rack` | `sim hang the mug on the mug rack` |

The training mixture remapped its sim task names, and prompts are derived from the name after
the remap, so the put-bottles episodes were labelled with the *throw* prompt.
The release ships an `abc_dit_xl_200k_model.json` sidecar
recording the map; `eval_policy.py` reads it back and prints the prompt it chose.
Keep the sidecar beside the `.pt` — which is what `--pretrained` does — and the
defaults are correct. This checkpoint's `max_action_prefix` is 8 during training.

Two caveats specific to it:

- The genuine throw-bottles sim scene (`--task bottles`) was left out of the
  training mixture as physically distinct from the real throw data, so expect
  roughly 1 in 4 there. Evaluate `put_plastic_bottles_in_bin` instead.

Read `mean_max_progress` from `summary.json` for numbers comparable to the
paper's sim-progress figures, and `success_rate` for the strict metric.

### Finetuned per-task checkpoints

Every sim task below has a checkpoint finetuned from `abc_dit_xl_200k_model.pt`
for 25k steps on that task's sim_224 episodes, published with its rollout
videos and eval stats. They are described by a checkpoint manifest
(`https://abc-data.timehorizons.org/finetuned_sim/manifest.json`): per task the
published steps with sizes and sha256, the recommended step (best combined
eval successes — not always 25k), and the exact prompt to evaluate under.
Download through it (no credentials needed; sha256-verified):

```bash
uv run prepare.py --sim-checkpoint-list       # what's published, with results
uv run prepare.py --sim-checkpoint pour       # the recommended step
uv run prepare.py --sim-checkpoint pour@25000 # a specific step
# Lands in cache/finetuned_sim/<task>/<step>.pt (~8 GB: model-only, all any
# eval or viz needs) and prints the exact eval command. Pass
# --sim-checkpoint-full-state for the ~24 GB training-state file (optimizer
# included, valid for --resume-from). Raw URLs follow
#   https://abc-data.timehorizons.org/finetuned_sim/<task>/checkpoints/<step>.pt
# with the model-only variant beside it at .../<step>_model.pt
```

Expected results, measured as strict `success_rate` over 20 randomized worlds
per seed (two seeds: 20260511 and 20260512, the default RTC evaluation, the
default 236-chunk budget), with `mean_max_progress` beside them. The prompt column is
the exact string the checkpoint trained under -- evaluate with it (for the
directive tasks the env rewrites the live prompt each reset, so any seed
prompt works):

| `<task>` (URL path) | trained + eval prompt | success (seed A / B) | progress |
| --- | --- | --- | --- |
| `pour` | `sim pouring beads` | 13/20 / 16/20 | 0.65 / 0.80 |
| `sim_inhand_transfer_the_item_to_other_side` | `sim inhand transfer the item to other side` | 18/20 / 18/20 | 0.96 / 0.97 |
| `put_relative` | per-episode directive | 5/20 / 10/20 | 0.73 / 0.83 |
| `grab_clutter` | per-episode directive | 5/20 / 3/20 | 0.25 / 0.15 |
| `conveyor_pick` | `conveyor pick` | 2/20 / 2/20 | 0.38 / 0.37 |
| `sim_sweep_away_paper_scraps_from_the_table` | `sim sweep away paper scraps from the table` | 2/20 / 1/20 | 0.26 / 0.14 |
| `count_into_opaque_box` | per-episode directive | 1/20 / 0/20 | 0.28 / 0.17 |
| `lego_blocks_sorting` | `lego blocks sorting` | 0/20 / 1/20 (400 chunks, see below) | 0.74 / 0.75 |
| `ball_tray_balancing` | `grab the tray handles and keep the ball balanced on the tray` | 1/20 / 0/20 (see below) | 1.00 / 1.00 |
| `nuts_bolts_sorting` | `nuts bolts sorting` | 0/20 / 0/20 | 0.08 / 0.12 |
| `sim_put_the_plastic_bottles_in_the_bin` | `sim put the plastic bottles in the bin` | **19/20 / 18/20 (20k)** | 0.99 / 0.95 |
| `sim_load_the_plates_into_the_dish_rack` | `sim load the plates into the dish rack` | **15/20 / 14/20 (20k)** | 0.85 / 0.87 |

put-bottles and dishrack peak at 20k (found by 5k-interval sweeps), so their
`checkpoints/20000.pt` is published beside `checkpoints/25000.pt` and is the
one to reach for.

Every checkpoint above is retrained on the v3 release
(the action-alignment fix documented in the release bucket's
`notes/action_alignment.md`); rows update as each task's retrain is evaluated,
and each task's superseded checkpoints move to `checkpoints_pre_v3/` alongside.
Rows before the conveyor retrain were measured under MuJoCo 3.6; the repo now
pins 3.8, and contact-heavy tasks are sensitive to the engine step (the
dishrack 20k checkpoint re-measures at 12/20 on seed A under 3.8).

Reading notes: ball_tray is the maintenance task -- every world earned full
hold-duration credit but the strict criterion asks the policy to hold roughly
4x longer than any demonstration, so its zero is a horizon artifact. lego is
the opposite case: 73% of its demonstrations are longer than the default
236-chunk budget, so its row is measured at `--num-chunks 400` (its published
eval summaries too); at the default budget every world times out mid-sort. The
inverted-convention and directive-relabel history behind nuts_bolts_sorting
and put_relative is documented in their evaluator docstrings.

How each training prompt is derived (and why eval must match it): an episode's
`episode_metadata.json` `prompt_timeline` wins when it has one, resolved per
frame -- `multi_drawer_search` advances through 2-4 targets mid-episode; else
`instruction` when that carries a real directive; otherwise the dataset
`task_name` with underscores as spaces. Two
historical traps worth knowing: the 200k parent pretrained the put-bottles
scene under the THROW wording (`sim throw plastic bottles in bin`) and
dishrack under `sim load plates into tabletop dish rack` -- those sidecar
prompts apply when evaluating the PARENT, while a checkpoint finetuned on the
sim_224 episodes trains and evaluates under the dataset-derived wording
(`sim put the plastic bottles in the bin`, `sim load the plates into the dish
rack`).

To reproduce a finetune end to end (any dataset task name from
`prepare.py --sim-data-list`):

```bash
# 1. Episodes + assets into a fresh per-task cache root, parent alongside.
uv run prepare.py --sim-data lego_blocks_sorting --cache /data/ft/lego/cache
uv run prepare.py --pretrained --cache /data/ft/lego/cache

# 2. 25k steps on 2 GPUs -- the recipe every published checkpoint used
#    (batch 90/GPU, LR 1e-4, checkpoints every 5k under finetune_checkpoints/).
torchrun --standalone --nproc-per-node 2 train.py \
    --cache-root /data/ft/lego/cache \
    --load-pretrained --mixture-preset sim_task \
    --flow.max-action-prefix 8 --train-steps 25000 \
    --val-every 5000 --val-batches 40 --ckpt-every 5000

# 3. Evaluate with the prompt the run trained under (printed at startup).
uv run eval_policy.py \
    --checkpoint /data/ft/lego/cache/finetune_checkpoints/25000.pt \
    --task lego_blocks_sorting --prompt "lego blocks sorting" \
    --num-worlds 20 --save-video --video-every-n-actions 15
```

### Matching the production eval numbers

The 236-chunk horizon matches the production sim-eval protocol. The tables in
this README are measured under the default RTC evaluation; production-era
dashboards used an older synchronous inference mode, so their absolute numbers
are not directly comparable (background: the release bucket's
notes/action_alignment.md).
Two reading notes when comparing against those numbers:

- `mean_max_progress` — mean over worlds of the best instantaneous progress
  fraction — is the statistic the production dishrack eval aggregated, and the
  right column to compare for any task where progress can be undone (a plate
  knocked back out, or still touching the gripper at the end, subtracts from
  `mean_reward` but not from `mean_max_progress`). `success_rate` is stricter
  than both: every object placed simultaneously, at any instant. On a
  maintenance task (one that reports `ever_failed`, like
  `ball_tray_balancing`) the reading flips: progress starts maximal and can
  only be lost, so `mean_max_progress` reads ≈1.0 regardless of when the ball
  drops — read `success_rate` there.
- Training remaps some sim task names before they become prompts, so two
  catalogue tasks read best under a prompt that differs from their spec
  default. For checkpoints trained on the xdof + sim mixtures: the put-bottles
  scene's sim data trained as `--prompt "sim throw plastic bottles in bin"`,
  and hang-mug as `--prompt "sim hang the mug on the mug rack"`. Mug flip,
  dishrack, and sweep defaults already match training.

## Assets

Large simulator meshes and textures are not committed to git. They ship as 37
per-package tarballs listed in `abc_sim/models/assets_manifest.json` (640.5 MB to
download, 1.7 GB unpacked) and install into `abc_sim/models/assets/`:

```bash
uv run prepare.py --sim                                   # every package
uv run prepare.py --sim-task load_plates_into_dish_rack   # one task's packages
uv run prepare.py --sim-list                              # sizes and file counts
```

`--sim-task` takes task names, aliases, or prompts — anything
`abc_sim.get_task_spec` resolves — and installs only the packages that task's
scene loads:

| Task | Packages | Download |
|---|---|---|
| `throw_plastic_bottles_in_bin` | `i2rt_yam` | 4.6 MB |
| `ball_tray_balancing` | `i2rt_yam` | 4.6 MB |
| `sort_lego_blocks` | `i2rt_yam`, `task_bins` | 4.7 MB |
| `sort_nuts_and_bolts` | `i2rt_yam`, `task_bins`, `task_nuts_bolts` | 4.9 MB |
| `sweep_away_paper_scraps_from_table` | `i2rt_yam`, `brush_flat`, `dustpan`, `garbage_can`, `paper_ball` | 8.5 MB |
| `turn_mug_right_side_up` | `i2rt_yam`, `mug`, `tray`, `task_mug_flip` | 18.1 MB |
| `multi_drawer_search` | `i2rt_yam`, `blocks`, `task_bins`, `task_multi_drawer_search` | 26.6 MB |
| `put_plastic_bottles_in_bin` | `i2rt_yam`, `task_water_bottles` | 45.7 MB |
| `load_plates_into_dish_rack` | `i2rt_yam`, `task_dishrack` | 58.2 MB |
| `grab_specific_object_from_clutter` | `i2rt_yam`, `task_bins`, `task_grab_clutter` | 72.9 MB |

Sizes are decimal MB, the same convention `--sim-list` prints. The table covers
the ten most-downloaded scenes; the other mapped tasks (`inhand_transfer`,
`conveyor_pick`, `count_into_opaque_box`, `put_relative`, `mug_tree`, `pour`,
`chess`, `blocks`) resolve the same way, with `inhand_transfer` additionally
needing the RoboCasa object packs described below. `task_bins`, the bin fixtures the search and sorting tasks
share, is a different package from `bin`, which no mapped task uses — the two are
easy to mix up once abbreviated, so the table spells them out.

`--sim` and `--sim-task` are a union, not a filter, so
`--sim --sim-task sort_lego_blocks` installs all 37 packages rather than that
task's 2. Pass `--sim-task` on its own to get the download in the table.

Every archive is checked against the manifest SHA256 before it is unpacked, and
each installed package records its hash under
`abc_sim/models/assets/.abc_sim_asset_packages/`, so reruns skip work that is
already done. Add `--sim-force` to reinstall, or `--sim-package <name> <name>`
to install packages by manifest name (space-separated; repeating the flag keeps
only the last name).

A package that cannot be fetched or fails its hash check does not stop the run:
the others still install, the closing summary lists each failure with its
reason, and the exit status is nonzero so a partial install is not mistaken for
a complete one. Re-running retries only what is missing.

To install from tarballs you already have (an offline mirror, a shared
filesystem, a package that is not on the mirror yet), point `--sim-source` at
the directory. Anything it does not carry still comes over https:

```bash
uv run prepare.py --sim --sim-source /path/to/archives
```

That fallback makes a partial directory useful — a handful of tarballs top up an
otherwise complete install without any extra flags — but it does mean
`--sim-source` is not an offline switch. Combined with `--sim` on a fresh clone,
a directory holding four archives installs those four and fetches the other 33
from the mirror. Name the packages you want with `--sim-package` if that is not
what you meant.

The `inhand_transfer` scene additionally needs RoboCasa's public object packs.
Those come from `utexas.box.com` rather than the ABC mirror and are a 2.8 GB
download (7.5 GB unpacked), so they are opt-in and `--sim` leaves them out:

```bash
uv run prepare.py --sim-robocasa
```

## Rendering

Live environment rendering is selected through `camera_backend`:

```python
env = abc_sim.make_env(
    task="put_bottles",
    render_cameras=True,
    camera_backend="mujoco",
    camera_height=168,
    camera_width=224,
)
```

Supported paths:

- `camera_backend="mujoco"`: standard MuJoCo renderer for local/single-env
  rendering. Requires a working OpenGL context.
- `camera_backend="mjwarp"`: GPU renderer used by policy evaluation.
- `camera_backend="blender"`: path-traced (Blender Cycles) renders of the same
  cameras, one Blender process per environment, for evaluating policies under a
  photoreal domain. About 190 ms per three-camera observation at 224x168 on an
  H100 (16 samples, denoised) plus ~4 s of startup per reset, so one world at a
  time. Needs a Blender 4.2+ binary (`BLENDER=/path/to/blender`), `usd-core` in
  the Python environment and optionally `ABC_HDRI` for a studio HDRI;
  `ABC_BLENDER_SAMPLES` / `ABC_BLENDER_DENOISER` trade quality for speed. See
  [abc_sim/rendering/blender/README.md](rendering/blender/README.md).
- `camera_backend="madrona"`: not supported in this release. The Madrona
  renderer needs a build that is not part of the public package, so use
  `mujoco` or `mjwarp` instead.

**Headless GPU boxes: check your EGL vendor.** Many cloud images ship the CUDA
userland without NVIDIA's OpenGL libraries, and `MUJOCO_GL=egl` then silently
falls back to Mesa's software rasterizer — everything works, just 10–40× slower
wherever the `mujoco` backend renders (we measured 8 vs 90+ frames/s). Check
`ls /usr/share/glvnd/egl_vendor.d/`: if there is no `10_nvidia.json`, install
the GL package matching `nvidia-smi`'s driver version, e.g.
`sudo apt-get install libnvidia-gl-580-server` for a 580.x server driver (no
reboot needed for headless EGL). On multi-GPU machines also set
`MUJOCO_EGL_DEVICE_ID` to the physical GPU index — EGL enumerates all GPUs
regardless of `CUDA_VISIBLE_DEVICES`.

The camera renderers behind these backends live under `abc_sim/rendering/`.

## Adding a New Env or Task

At minimum, a new task needs:

1. A MuJoCo scene XML under `abc_sim/models/`.
2. Any required meshes/textures under `abc_sim/models/assets/`.
3. A `SceneTaskSpec` entry in `abc_sim/task_registry.py`.
4. A `SimTaskSpec` entry in `abc_sim/task_specs.py`.

Most tasks should also add:

- A randomizer under `abc_sim/randomization/tasks/`, then register it in
  `abc_sim/randomization/registry.py`.
- A task evaluator under `abc_sim/task_eval/`, then register it in
  `abc_sim/task_eval/registry.py`.
- Runtime hooks under `abc_sim/task_runtime/` if the environment has dynamic
  behavior beyond ordinary MuJoCo stepping.
- Asset package metadata in `abc_sim/models/assets_manifest.json` if the new
  task needs large mesh or texture files, plus an entry in `prepare.py`'s
  `SIM_TASK_PACKAGES` map so `--sim-task` can install them.

Keep simulator-owned behavior inside `abc_sim`. Teleop/data-collection logic
belongs outside this package.

## RL Usage

For RL, use one-action stepping and Gymnasium termination semantics:

```python
import abc_sim

env = abc_sim.make_env(
    task="put_bottles",
    render_cameras=False,
    max_episode_steps=1000,
    terminate_on_success=True,
)

obs, reset_info = env.reset(seed=0, randomize=True)
done = False
while not done:
    action = policy(obs)  # shape: (14,)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

Use `info["task_success"]` and `info["task_eval"]` for success-conditioned
metrics, curricula, or offline analysis. Use `reset_info["randomization"]` to
record and replay fixed worlds across policies.
