# /// script
# requires-python = ">=3.10"
# dependencies = ["tyro"]
# ///
"""Download and unpack the public abc bottles-in-bin dataset and abc_sim assets."""

import hashlib
import http.client
import json
import shlex
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro

from abc_minimal.config import default_cache_root

DATA_BASE = "https://abc-data.timehorizons.org"
DEFAULT_CACHE = default_cache_root()
# Load-bearing: the data mirror answers 403 to the default "Python-urllib/*" agent, which
# looks exactly like an unpublished package. Any other value works; do not drop the header.
HTTP_HEADERS = {"User-Agent": "abc-prepare/1.0"}
REPO_ROOT = Path(__file__).resolve().parent

SMALL_FILES = [
    ("misc/norm_stats.json",                                "norm_stats.json"),
]

PREVIEW_TAR = "dataset/dataset_preview/bottles_in_bin_preview.tar"
FULL_TARS = [
    "dataset/dataset_preview/bottles_in_bin_real.tar",
    "dataset/dataset_preview/bottles_in_bin_sim.tar",
]

# Pretrained 75k from-scratch bottles policy (model-only, fp32 weights, ~8.1 GB).
# Lands at cache/bottles_75k.pt by default; override with ABC_CACHE.
CHECKPOINT_KEY = "checkpoints/bottles_release_prep_75k.pt"
CHECKPOINT_DST = "bottles_75k.pt"

# Multi-task sim DiT-XL parent, step 200k (model-only, fp32 weights, ~8.1 GB): the
# checkpoint --load-pretrained finetunes from, and the one to evaluate the five sim
# tasks with. Its sidecar carries the sha256 verified below and the per-task training
# prompts eval_policy.py reads back, so the two are downloaded together and the
# filenames must keep the same stem. PRETRAINED_DST matches TrainConfig's
# pretrained_ckpt_name default, so --load-pretrained needs no flag.
PRETRAINED_KEY = "checkpoints/abc_dit_xl_200k_model.pt"
PRETRAINED_META_KEY = "checkpoints/abc_dit_xl_200k_model.json"
PRETRAINED_DST = "abc_dit_xl_200k_model.pt"
PRETRAINED_META_FORMAT = "abc_checkpoint_metadata/v1"

# Real-robot DAgger finetunes of ABC-DiT (model-only, fp32 weights, ~8.1 GB each).
# Each <name>.json sidecar carries the sha256 and the deploy flags it was run with.
DAGGER_KEY_PREFIX = "checkpoints/dagger"
DaggerCheckpoint = Literal[
    "folding_paper_box",
    "insert_earbuds_1eb",
    "insert_earbuds_3eb",
    "using_a_key",
    "pack_student_bag",
]

# Released weights-only VLA parents. The default is the abc130k 200k checkpoint:
# it shares the released DiT parent's five-task sim mixture, while vla_200k was
# trained on the earlier xdof-only mixture and therefore has no declared abc_sim
# task bundle. Every selected checkpoint is downloaded with its metadata sidecar
# and stored under a unique flat cache name suitable for --pretrained-ckpt-name.
VLA_PRETRAINED_FAMILIES = {
    "abc130k": {
        "key_prefix": "checkpoints/vla_abc130k_v2",
        "dst_prefix": "vla_abc130k",
        # The VLA sidecars predate sim_prompt_map. This is the authoritative
        # sidecar for the identical training mixture and supplies the exact five
        # task prompts until the VLA release metadata itself carries the map.
        "sim_prompt_meta_key": PRETRAINED_META_KEY,
        "training_mixture": "xdof_csf_3_5k_may20_sim0514_hours_weighted",
    },
    "200k": {
        "key_prefix": "checkpoints/vla_200k_v2",
        "dst_prefix": "vla_200k",
        "sim_prompt_meta_key": None,
        "training_mixture": "xdof",
    },
}

# Simulator meshes/textures are not committed to git; they ship as tarballs listed in
# abc_sim/models/assets_manifest.json and unpack into the (gitignored) abc_sim/models tree.
SIM_MANIFEST = REPO_ROOT / "abc_sim" / "models" / "assets_manifest.json"
SIM_MANIFEST_FORMAT = "abc_sim_assets_manifest/v1"
SIM_MODELS_DIR = REPO_ROOT / "abc_sim" / "models"
# One marker dir for every package, keyed by package name. It lives under assets/ because
# that path is already gitignored.
SIM_MARKER_DIR = SIM_MODELS_DIR / "assets" / ".abc_sim_asset_packages"

# Every scene loads the YAM arm meshes; everything else is per-task. Scene names are the
# `env_task` values from abc_sim/task_specs.py; user-facing task names resolve onto them.
SIM_BASE_PACKAGES = ("i2rt_yam",)
SIM_TASK_PACKAGES = {
    "bottles": (),
    "put_bottles": ("task_water_bottles",),
    "dishrack": ("task_dishrack",),
    "mug_flip": ("mug", "tray", "task_mug_flip"),
    "sweep": ("brush_flat", "dustpan", "garbage_can", "paper_ball"),
    "inhand_transfer": ("task_inhand_transfer", "assets_robocasa"),
    "grab_clutter": ("task_grab_clutter", "task_bins"),
    "multi_drawer_search": ("task_multi_drawer_search", "task_bins", "blocks"),
    "nuts_bolts_sorting": ("task_nuts_bolts", "task_bins"),
    "lego_blocks_sorting": ("task_bins",),
    # Tray and ball are primitive geoms in the scene XML; the arm meshes are enough.
    "ball_tray_balancing": (),
    # Several of the task_* packages below are pulled in by abc_sim.randomization rather
    # than the scene XML, so only reset(randomize=True) shows them missing.
    "conveyor_pick": ("task_conveyor_pick",),
    # The box and the counted cubes are primitive geoms; the arm meshes are enough.
    "count_into_opaque_box": (),
    "stack_tallest_tower": (),
    "put_relative": ("task_put_relative",),
    "mug_tree": ("mug", "mug_tree", "task_mug_tree"),
    "pour": ("cup_stacking", "mug"),
    "chess": ("chess", "task_chess"),
    "blocks": ("blocks",),
}
# Policy-specific VLA releases use a distinct catalogue prefix so they cannot
# replace the existing DiT checkpoint for the same simulator task.
VLA_SIM_CHECKPOINT_PREFIX = "vla_"
# Episode data for the 224x224 sim release, one tarball per dataset task_name. The task
# names are the dataset's, not abc_sim's: 8 of the 24 carry no sim_ prefix, and the
# spelling scene ships as seven separate tasks.
DATA_MANIFEST_KEY = "dataset/sim_224/manifest.json"
DATA_MANIFEST_FORMAT = "abc_sim_dataset_manifest/v1"
DATA_MANIFEST_CACHE = "sim_224_manifest.json"
# Finetuned per-task checkpoints, described by their own manifest: per task the
# published steps (uri/bytes/sha256), the recommended step (best combined eval
# successes across both seeds), and the exact prompt to evaluate under.
CKPT_MANIFEST_KEY = "finetuned_sim/manifest.json"
CKPT_MANIFEST_FORMAT = "abc_finetuned_sim_manifest/v1"
CKPT_MANIFEST_CACHE = "finetuned_sim_manifest.json"
SIM_DATA_TAR_DIR = "sim_224_tars"
SIM_DATA_MARKER_DIR = ".sim_data_packages"
SCENE_DATASET_TASKS = {
    "ball_tray_balancing": ("ball_tray_balancing",),
    "conveyor_pick": ("conveyor_pick",),
    "count_into_opaque_box": ("count_into_opaque_box",),
    "grab_clutter": ("grab_clutter",),
    "lego_blocks_sorting": ("lego_blocks_sorting",),
    "multi_drawer_search": ("multi_drawer_search",),
    "nuts_bolts_sorting": ("nuts_bolts_sorting",),
    "put_relative": ("put_relative",),
    "mug_tree": ("sim_hang_the_mug_on_the_mug_rack",),
    "inhand_transfer": ("sim_inhand_transfer_the_item_to_other_side",),
    "dishrack": ("sim_load_the_plates_into_the_dish_rack",),
    "pour": ("sim_pouring_beads",),
    "put_bottles": ("sim_put_the_plastic_bottles_in_the_bin",),
    "chess": ("sim_set_up_chess_pieces_on_the_board",),
    "blocks": (
        "sim_spell_abc",
        "sim_spell_agi",
        "sim_spell_cat",
        "sim_spell_dog",
        "sim_spell_fish",
        "sim_spell_xdof",
        "sim_spell_yam",
    ),
    "sweep": ("sim_sweep_away_paper_scraps_from_the_table",),
    "bottles": ("sim_throw_plastic_bottles_in_bin",),
    "mug_flip": ("sim_turn_the_mug_right_side_up",),
}
# Reverse of the above. Lets an exact dataset task name find its scene (and so its asset
# packages) without the manifest, which matters before the release is published.
DATASET_TASK_SCENES = {
    task: scene for scene, tasks in SCENE_DATASET_TASKS.items() for task in tasks
}

# assets_robocasa ships only a fetch script; the object packs themselves come straight from
# RoboCasa's public Box mirror, so they are opt-in. Measured 2026-08-12: 2.8 GB of zips that
# unpack to 7.5 GB (the vendored download_assets.sh still claims ~220 MB).
ROBOCASA_PACKAGE = "assets_robocasa"
ROBOCASA_DIR = SIM_MODELS_DIR / "assets_robocasa"
ROBOCASA_PACKS = (
    ("objects_lightwheel", "vckqvvkh1z8t69k8qcpcmee6k66stii4"),
    ("objaverse", "03eionyo8fk3a9dsksq9jb8du5lqfw8h"),
)
ROBOCASA_URL = "https://utexas.box.com/shared/static/{pack_id}.zip"


@dataclass
class PrepareConfig:
    """Download and unpack the public abc bottles-in-bin dataset and abc_sim assets."""

    full: Annotated[
        bool,
        tyro.conf.arg(help="Download the 35 GB real + sim tars instead of the preview tar."),
    ] = False
    skip_extract: Annotated[
        bool,
        tyro.conf.arg(help="Leave tars on disk without extracting them."),
    ] = False
    checkpoint: Annotated[
        bool,
        tyro.conf.arg(help="Download the 8.1 GB pretrained 75k bottles policy."),
    ] = False
    pretrained: Annotated[
        bool,
        tyro.conf.arg(
            help="Download the 8.1 GB multi-task sim DiT-XL parent (200k) and its "
            "metadata sidecar, plus assets for every supported sim task. Skips "
            "episode data unless --full or --checkpoint."
        ),
    ] = False
    vla_pretrained: Annotated[
        bool,
        tyro.conf.arg(
            help="Download a released 8.8 GB VLA parent and metadata sidecar, "
            "sha256-verify it, and install assets for its declared sim tasks. "
            "Defaults to abc130k at step 200000."
        ),
    ] = False
    vla_pretrained_family: Annotated[
        Literal["abc130k", "200k"],
        tyro.conf.arg(
            help="VLA release family selected by --vla-pretrained. abc130k is the "
            "recommended sim-finetuning parent; 200k is the earlier xdof-only run."
        ),
    ] = "abc130k"
    vla_pretrained_step: Annotated[
        Literal[50000, 100000, 200000],
        tyro.conf.arg(help="Published VLA step selected by --vla-pretrained."),
    ] = 200000
    dagger_checkpoint: Annotated[
        tuple[DaggerCheckpoint, ...],
        tyro.conf.arg(
            help="Download these real-robot DAgger checkpoints (~8.1 GB each, "
            "sha256-verified) and print their deploy commands. Skips the dataset."
        ),
    ] = ()
    cache: Annotated[
        Path,
        tyro.conf.arg(help=f"Where to put files; defaults to ABC_CACHE or {DEFAULT_CACHE}."),
    ] = DEFAULT_CACHE
    sim: Annotated[
        bool,
        tyro.conf.arg(help="Install every abc_sim asset package (640.5 MB). Skips the dataset."),
    ] = False
    sim_task: Annotated[
        tuple[str, ...],
        tyro.conf.arg(
            help="Install just the assets these sim tasks need, space-separated. "
            "Skips the dataset."
        ),
    ] = ()
    sim_bundle: Annotated[
        tuple[str, ...],
        tyro.conf.arg(
            help="Install the assets and recommended finetuned checkpoint for these "
            "sim tasks, space-separated. Does not download episode data."
        ),
    ] = ()
    sim_bundle_list: Annotated[
        bool,
        tyro.conf.arg(help="List the published task bundles and exit."),
    ] = False
    sim_package: Annotated[
        tuple[str, ...],
        tyro.conf.arg(
            help="Install asset packages by manifest name, space-separated "
            "(--sim-package mug tray). Repeating the flag keeps only the last name. "
            "See --sim-list."
        ),
    ] = ()
    sim_source: Annotated[
        Path | None,
        tyro.conf.arg(
            help="Read asset tarballs from this local directory, falling back to https "
            "for any the directory does not carry."
        ),
    ] = None
    sim_robocasa: Annotated[
        bool,
        tyro.conf.arg(
            help="Also fetch the RoboCasa object packs from utexas.box.com "
            "(2.8 GB download, 7.5 GB unpacked). Only the inhand_transfer scene needs them."
        ),
    ] = False
    sim_force: Annotated[
        bool,
        tyro.conf.arg(help="Re-download and re-extract packages that are already installed."),
    ] = False
    sim_list: Annotated[
        bool,
        tyro.conf.arg(help="List the asset packages in the manifest and exit."),
    ] = False
    sim_data: Annotated[
        tuple[str, ...],
        tyro.conf.arg(
            help="Download these tasks' episode data from the public dataset/sim_224 "
            "release into the cache, space-separated. Accepts the same names --sim-task "
            "accepts plus exact dataset task names, and also installs their assets. "
            "See --sim-data-list."
        ),
    ] = ()
    sim_data_list: Annotated[
        bool,
        tyro.conf.arg(help="List the data tasks in the published sim_224 manifest and exit."),
    ] = False
    sim_checkpoint: Annotated[
        tuple[str, ...],
        tyro.conf.arg(
            help="Download finetuned per-task checkpoint(s) by published task name, "
            "optionally name@step (default: the manifest's recommended step). "
            "sha256-verified when the manifest carries a hash. "
            "See --sim-checkpoint-list."
        ),
    ] = ()
    sim_checkpoint_list: Annotated[
        bool,
        tyro.conf.arg(help="List the published finetuned checkpoints and exit."),
    ] = False
    sim_checkpoint_full_state: Annotated[
        bool,
        tyro.conf.arg(
            help="Download the ~24 GB full training-state checkpoint (optimizer "
            "included, for --resume-from) instead of the ~8 GB model-only file."
        ),
    ] = False


def _fmt_bytes(n):
    # Use decimal units to match --sim-list and manifest output.
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1000


def _fmt_eta(sec):
    if sec is None or sec == float("inf"):
        return "--:--"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(frac, width=24):
    filled = int(width * frac)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _progress(prefix, done, total, start, *, end=False):
    """Render one-line progress to stderr with carriage return, clamped to terminal width."""
    elapsed = max(time.monotonic() - start, 1e-6)
    rate = done / elapsed
    if total:
        frac = min(done / total, 1.0)
        eta = (total - done) / rate if rate > 0 else None
        line = f"{prefix} {_bar(frac)} {frac*100:5.1f}%  {_fmt_bytes(done)}/{_fmt_bytes(total)}  {_fmt_bytes(rate)}/s  ETA {_fmt_eta(eta)}"
    else:
        line = f"{prefix} {_fmt_bytes(done)}  {_fmt_bytes(rate)}/s"
    cols = shutil.get_terminal_size((100, 24)).columns
    if len(line) > cols - 1:
        line = line[: max(cols - 1, 1)]
    sys.stderr.write("\r\x1b[2K" + line)
    if end:
        sys.stderr.write("\n")
    sys.stderr.flush()


def remote_size(url):
    """Return Content-Length from a HEAD, or None if the object is missing."""
    req = urllib.request.Request(url, method="HEAD", headers=HTTP_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_length = resp.headers.get("Content-Length")
            return int(content_length) if content_length is not None else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def download(key, dst, max_attempts=5):
    """Download key to dst with a progress bar. Idempotent on full-size files."""
    download_url(f"{DATA_BASE}/{key}", dst, max_attempts=max_attempts)


def download_url(
    url, dst, *, expected=None, require_size=True, max_attempts=5, headers=None, label=None
):
    """Download URL to dst, retrying and resuming verified partial bodies.

    ``expected`` skips the HEAD request when the size is already known. Hosts that
    reject HEAD (the RoboCasa Box mirror) need ``require_size=False``, which streams
    with an open-ended progress line.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if expected is None:
        try:
            expected = remote_size(url)
        except Exception:
            if require_size:
                raise
            expected = None
        if expected is None and require_size:
            raise RuntimeError(f"object not found: {url}")
    label = label or dst.name
    if expected and dst.exists() and dst.stat().st_size == expected:
        print(f"[skip] {label} ({expected/1e6:.1f} MB)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    prefix = f"[get] {label}"
    transfer_start = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        # With a known total, resume a valid partial body. For an open-ended
        # response there is no reliable way to distinguish a complete .part
        # from a truncated one, so restart it.
        done = tmp.stat().st_size if tmp.exists() and expected is not None else 0
        if expected is None and tmp.exists():
            tmp.unlink()
        elif tmp.exists() and done == expected:
            _progress(prefix, done, expected, transfer_start, end=True)
            tmp.replace(dst)
            return
        elif expected is not None and done > expected:
            tmp.unlink()
            done = 0

        request_headers = {**HTTP_HEADERS, **(headers or {})}
        mode = "wb"
        if done:
            request_headers["Range"] = f"bytes={done}-"
            mode = "ab"
        req = urllib.request.Request(url, headers=request_headers)
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            if done:
                content_range = resp.headers.get("Content-Range", "")
                expected_prefix = f"bytes {done}-"
                if resp.status != 206 or not content_range.startswith(expected_prefix):
                    # The server ignored or mishandled Range. Restart this
                    # attempt from zero instead of appending an invalid body.
                    resp.close()
                    done = 0
                    mode = "wb"
                    resp = urllib.request.urlopen(
                        urllib.request.Request(url, headers={**HTTP_HEADERS, **(headers or {})}),
                        timeout=300,
                    )

            if expected is None:
                response_size = int(resp.headers.get("Content-Length") or 0)
                expected = response_size or None

            with resp, open(tmp, mode) as out:
                done = _stream(
                    resp, out, done, expected, prefix, transfer_start
                )
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                tmp.unlink(missing_ok=True)
            elif isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code not in (408, 429):
                raise  # auth/not-found are not transient
            if attempt == max_attempts:
                raise
            print(f"\n[retry {attempt}/{max_attempts}] {label}: {e}; "
                  f"resuming from {tmp.stat().st_size if tmp.exists() else 0} bytes")
            continue
        actual = tmp.stat().st_size
        if expected is None or actual == expected:
            _progress(prefix, actual, expected, transfer_start, end=True)
            tmp.replace(dst)
            return
        if attempt == max_attempts:
            raise RuntimeError(
                f"{label}: got {actual} bytes, expected {expected} after "
                f"{max_attempts} attempts"
            )
        print(f"\n[retry {attempt}/{max_attempts}] {label}: got {actual}/{expected} "
              "bytes; resuming")


def _stream(resp, out, done, expected, prefix, start):
    """Copy an HTTP response body to an open file, updating the progress bar."""
    last = 0.0
    chunk = 1 << 20
    while True:
        buf = resp.read(chunk)
        if not buf:
            break
        out.write(buf)
        done += len(buf)
        now = time.monotonic()
        if now - last >= 0.25:
            _progress(prefix, done, expected, start)
            last = now
    return done


def extract_tar(tar_path, cache, *, force_role=None):
    """Untar bottles_in_bin_*.tar into the cache dir with a progress bar.

    Tar layout: ``train/<eid>/...``, ``val/<eid>/...``. Each episode's
    metadata.json is read first; episodes whose ``task_name`` starts with
    ``sim_`` go to ``{train,val}_sim/``, the rest to ``{train,val}_real/``.

    ``force_role`` skips that classification pass and routes every episode to
    ``{train,val}_<force_role>/``. The sim_224 tars need it: 8 of the 24 dataset
    task_names carry no ``sim_`` prefix, so sniffing would file them as real.
    """
    print(f"[extract] {tar_path.name}")
    total_size = tar_path.stat().st_size

    eid_role = {}  # (split, eid) -> "real" | "sim"
    if force_role is None:
        pre_prefix = f"[scan] {tar_path.name}"
        pre_start = time.monotonic()
        pre_last = 0.0
        with open(tar_path, "rb") as raw, tarfile.open(fileobj=raw) as tf:
            for member in tf:
                now = time.monotonic()
                if now - pre_last >= 0.25:
                    _progress(pre_prefix, raw.tell(), total_size, pre_start)
                    pre_last = now
                if not member.isfile() or Path(member.name).name != "episode_metadata.json":
                    continue
                parts = Path(member.name).parts
                if len(parts) < 3 or parts[0] not in ("train", "val"):
                    continue
                with tf.extractfile(member) as f:
                    meta = json.loads(f.read())
                role = "sim" if str(meta.get("task_name", "")).startswith("sim_") else "real"
                eid_role[(parts[0], parts[1])] = role
        _progress(pre_prefix, total_size, total_size, pre_start, end=True)

    prefix = f"[untar] {tar_path.name}"
    start = time.monotonic()
    last = 0.0
    with open(tar_path, "rb") as raw, tarfile.open(fileobj=raw) as tf:
        for member in tf:
            now = time.monotonic()
            if now - last >= 0.25:
                _progress(prefix, raw.tell(), total_size, start)
                last = now
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2 or parts[0] not in ("train", "val"):
                continue
            role = force_role or eid_role.get((parts[0], parts[1]))
            if role is None:
                continue
            dst = cache / f"{parts[0]}_{role}" / Path(*parts[1:])
            if dst.exists() and dst.stat().st_size == member.size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, open(dst, "wb") as out:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
    _progress(prefix, total_size, total_size, start, end=True)


def fetch_tars(cache, tar_keys, skip_extract):
    paths = []
    for key in tar_keys:
        dst = cache / Path(key).name
        download(key, dst)
        paths.append(dst)
    if skip_extract:
        print("[skip-extract] tars left unextracted")
        return
    for p in paths:
        extract_tar(p, cache)


def fetch_pretrained(cache, config: PrepareConfig):
    """Download the sim DiT-XL parent, verified against its published sidecar.

    The sidecar comes first: it carries the size and digest, so a truncated or
    stale 8.1 GB body is caught here rather than at load time as a wrong-looking
    policy. eval_policy.py reads the same file back for its per-task prompts,
    which is why it lands beside the weights under the matching stem.
    """
    meta_dst = cache / Path(PRETRAINED_META_KEY).name
    print(f"[pretrained] {PRETRAINED_META_KEY}")
    download(PRETRAINED_META_KEY, meta_dst)
    meta = json.loads(meta_dst.read_text())
    if meta.get("format") != PRETRAINED_META_FORMAT:
        raise RuntimeError(
            f"{meta_dst.name}: expected format {PRETRAINED_META_FORMAT}, "
            f"got {meta.get('format')!r}"
        )

    tasks = tuple(meta.get("sim_prompt_map", {}))
    if tasks:
        asset_config = PrepareConfig(
            sim_task=tasks,
            sim_source=config.sim_source,
            sim_force=config.sim_force,
        )
        if prepare_sim(asset_config):
            raise SystemExit(1)

    dst = cache / PRETRAINED_DST
    print(f"[pretrained] {PRETRAINED_KEY} ({meta['bytes']/1e9:.1f} GB)")
    download_url(f"{DATA_BASE}/{PRETRAINED_KEY}", dst, expected=meta["bytes"])
    verify_sha256(dst, meta["sha256"], PRETRAINED_DST)
    print(f"[pretrained] sha256 ok, step {meta['step']}, licence {meta['license']}")
    # Print the licence qualification alongside its SPDX name.
    for note in meta.get("license_notes", ()):
        print(f"[pretrained] {note}")
    if tasks:
        try:
            display_dst = dst.relative_to(REPO_ROOT)
        except ValueError:
            display_dst = dst
        print(f"[pretrained] supports {len(tasks)} sim task(s):")
        for task in tasks:
            print(f"  {task}")
        print("[pretrained] replace TASK below with one of the task names above:")
        print(
            f"[eval] uv run eval_policy.py --checkpoint {display_dst} --task TASK "
            "--num-worlds 20"
        )
        print(
            f"[viz] uv run viz_policy.py --sim.checkpoint {display_dst} --sim.task TASK"
        )


def _load_checkpoint_metadata(key, dst):
    """Download and validate one published checkpoint metadata sidecar."""
    print(f"[metadata] {key}")
    download(key, dst)
    meta = json.loads(dst.read_text())
    if meta.get("format") != PRETRAINED_META_FORMAT:
        raise RuntimeError(
            f"{dst.name}: expected format {PRETRAINED_META_FORMAT}, "
            f"got {meta.get('format')!r}"
        )
    for field in ("uri", "bytes", "sha256", "step"):
        if field not in meta:
            raise RuntimeError(f"{dst.name}: checkpoint metadata has no {field!r} field")
    return meta


def _vla_sim_prompt_map(cache, release, meta):
    """Return the exact prompt map for a VLA release's declared sim mixture.

    New VLA metadata can carry the map directly. The current abc130k sidecars
    identify the same named mixture as the DiT parent but predate its
    sim_prompt_map field, so read that map from the DiT sidecar after first
    checking the VLA mixture identity. The xdof-only family deliberately has no
    fallback: it did not train on the five-task sim addition.
    """
    prompt_map = meta.get("sim_prompt_map")
    if prompt_map is not None:
        if not isinstance(prompt_map, dict):
            raise RuntimeError("VLA checkpoint sim_prompt_map must be a JSON object")
        return prompt_map

    prompt_meta_key = release["sim_prompt_meta_key"]
    if prompt_meta_key is None:
        return {}
    if meta.get("training_mixture") != release["training_mixture"]:
        raise RuntimeError(
            "refusing to borrow sim prompts for an unexpected VLA training mixture: "
            f"{meta.get('training_mixture')!r}"
        )

    prompt_meta_dst = cache / Path(prompt_meta_key).name
    prompt_meta = _load_checkpoint_metadata(prompt_meta_key, prompt_meta_dst)
    prompt_map = prompt_meta.get("sim_prompt_map")
    if not isinstance(prompt_map, dict) or not prompt_map:
        raise RuntimeError(f"{prompt_meta_dst.name}: missing non-empty sim_prompt_map")
    return prompt_map


def fetch_vla_pretrained(cache, config: PrepareConfig):
    """Download one released VLA parent and its matching prompt sidecar."""
    family = config.vla_pretrained_family
    step = config.vla_pretrained_step
    release = VLA_PRETRAINED_FAMILIES[family]
    key_prefix = release["key_prefix"]
    stem = f"{release['dst_prefix']}_{step}_v2"
    dst = cache / f"{stem}.pt"
    meta_dst = cache / f"{stem}.json"
    meta_key = f"{key_prefix}/{step}.json"

    meta = _load_checkpoint_metadata(meta_key, meta_dst)
    expected_uri = f"{DATA_BASE}/{key_prefix}/{step}.pt"
    if meta["uri"] != expected_uri:
        raise RuntimeError(
            f"{meta_dst.name}: expected checkpoint URI {expected_uri!r}, "
            f"got {meta['uri']!r}"
        )
    if int(meta["step"]) != step:
        raise RuntimeError(
            f"{meta_dst.name}: expected release step {step}, got {meta['step']!r}"
        )

    prompt_map = _vla_sim_prompt_map(cache, release, meta)
    if prompt_map and "sim_prompt_map" not in meta:
        # eval_policy.py reads the sidecar next to the locally renamed weights.
        # Record where the inherited map came from rather than making that
        # provenance implicit.
        meta["sim_prompt_map"] = prompt_map
        meta["sim_prompt_map_source"] = release["sim_prompt_meta_key"]
        meta["name"] = dst.name
        tmp = meta_dst.with_name(meta_dst.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n")
        tmp.replace(meta_dst)

    tasks = tuple(prompt_map)
    if tasks:
        asset_config = PrepareConfig(
            sim_task=tasks,
            sim_source=config.sim_source,
            sim_force=config.sim_force,
        )
        if prepare_sim(asset_config):
            raise SystemExit(1)

    print(f"[pretrained-vla] {meta['uri']} ({meta['bytes']/1e9:.1f} GB)")
    download_url(meta["uri"], dst, expected=meta["bytes"])
    verify_sha256(dst, meta["sha256"], dst.name)
    print(
        f"[pretrained-vla] sha256 ok, family {family}, release step {step}, "
        f"licence {meta.get('license', '?')}"
    )
    for note in meta.get("license_notes", ()):
        print(f"[pretrained-vla] {note}")

    try:
        display_dst = dst.relative_to(REPO_ROOT)
    except ValueError:
        display_dst = dst
    print(
        "[train] uv run train.py --policy vla "
        f"--cache-root {shlex.quote(str(cache))} --load-pretrained "
        f"--pretrained-ckpt-name {shlex.quote(dst.name)}"
    )
    if tasks:
        print(f"[pretrained-vla] supports {len(tasks)} sim task(s) from its training mixture:")
        for task in tasks:
            print(f"  {task}")
        print("[pretrained-vla] replace TASK below with one of the task names above:")
        print(
            f"[eval] uv run eval_policy.py --checkpoint {display_dst} --task TASK "
            "--num-worlds 20"
        )
        print(f"[viz] uv run viz_policy.py --sim.checkpoint {display_dst} --sim.task TASK")
    else:
        print(
            "[pretrained-vla] no abc_sim task bundle is declared for this "
            f"{meta.get('training_mixture', family)!r} training mixture"
        )


def _deploy_flag(key, value):
    flag = "--" + key.replace("_", "-")
    if value is True:
        return flag
    return f"{flag}={shlex.quote(json.dumps(value) if isinstance(value, list) else str(value))}"


def fetch_dagger_checkpoints(cache, names):
    """Download DAgger checkpoints and print the deploy command each was run with."""
    for name in names:
        dst = cache / "dagger" / f"{name}.pt"
        meta = _load_checkpoint_metadata(
            f"{DAGGER_KEY_PREFIX}/{name}.json", dst.with_suffix(".json")
        )
        print(f"[dagger] {meta['uri']} ({meta['bytes']/1e9:.1f} GB)")
        download_url(meta["uri"], dst, expected=meta["bytes"])
        verify_sha256(dst, meta["sha256"], dst.name)
        print(f"[dagger] sha256 ok, step {meta['step']}, licence {meta['license']}")
        try:
            display_dst = dst.relative_to(REPO_ROOT)
        except ValueError:
            display_dst = dst
        flags = " ".join(_deploy_flag(k, v) for k, v in meta["deploy"].items())
        print(f"[deploy] uv run deploy/deploy_policy.py --checkpoint-path {display_dst} {flags}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path, expected, label):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label}: sha256 mismatch for {path}\n"
            f"  expected {expected}\n"
            f"  actual   {actual}"
        )


def check_members(dest, names):
    """Reject archive members whose path would escape dest."""
    root = dest.resolve()
    for name in names:
        target = (dest / name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"refusing to extract {name!r} outside {dest}")


def extract_archive(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        check_members(dest, [m.name for m in tar.getmembers()])
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def load_sim_manifest():
    """Return manifest packages keyed by name, in manifest order."""
    manifest = json.loads(SIM_MANIFEST.read_text())
    if manifest.get("format") != SIM_MANIFEST_FORMAT:
        raise RuntimeError(
            f"unexpected manifest format in {SIM_MANIFEST}: {manifest.get('format')!r}"
        )
    return {pkg["name"]: pkg for pkg in manifest["packages"]}


def sim_installed(pkg):
    """True when the marker matches the manifest sha256 and the unpacked dirs are still there."""
    marker = SIM_MARKER_DIR / f"{pkg['name']}.sha256"
    if not marker.exists() or marker.read_text().strip() != pkg["sha256"]:
        return False
    dest = REPO_ROOT / pkg["extract_dir"]
    return all((dest / entry).exists() for entry in pkg["contains"])


def sim_archive(pkg, source, work_dir):
    """Return a sha256-verified archive for pkg, preferring source over its https uri."""
    name = pkg["name"]
    if source is not None:
        archive = source / pkg["archive"]
        if archive.exists():
            print(f"[local] {archive.name}")
            verify_sha256(archive, pkg["sha256"], name)
            return archive
        # A partial --sim-source is the normal case: it is how packages arrive before they
        # are published. Fetch whatever the directory does not carry.
        print(f"[fetch] {pkg['archive']} is not in {source}")
    archive = work_dir / pkg["archive"]
    if not (archive.exists() and sha256_file(archive) == pkg["sha256"]):
        archive.unlink(missing_ok=True)
        download_url(pkg["uri"], archive, expected=pkg["size_bytes"])
        verify_sha256(archive, pkg["sha256"], name)
    return archive


def install_sim_package(pkg, source, work_dir):
    """Fetch, verify and unpack one package, then mark it installed. Raises on any failure."""
    archive = sim_archive(pkg, source, work_dir)
    extract_archive(archive, REPO_ROOT / pkg["extract_dir"])
    marker = SIM_MARKER_DIR / f"{pkg['name']}.sha256"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(pkg["sha256"] + "\n")
    if archive.parent == work_dir:
        archive.unlink(missing_ok=True)


def sim_failure_reason(exc):
    """One-line reason for the summary table. A 404 usually means 'not published yet'."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason} <{exc.url}>"
    if isinstance(exc, urllib.error.URLError):
        return f"unreachable: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def print_sim_summary(
    installed,
    skipped,
    failed,
    *,
    label="sim",
    noun="package",
    consequence="any scene that needs their meshes will fail to load",
):
    """Report every package once. A single unpublished tarball must not hide in scrollback."""
    print(
        f"\n[{label}] {len(installed)} installed, {len(skipped)} already present, "
        f"{len(failed)} failed"
    )
    if not failed:
        return
    width = max(len(name) for name, _ in failed)
    for name, reason in failed:
        print(f"  FAILED  {name:{width}s}  {reason}")
    print(
        f"[error] {len(failed)} {noun}(s) did not install; {consequence}. "
        "Re-run to retry just those."
    )


def fetch_robocasa_packs():
    """Fetch the public RoboCasa object packs that assets_robocasa/download_assets.sh pulls."""
    ROBOCASA_DIR.mkdir(parents=True, exist_ok=True)
    pending = [pack for pack, _ in ROBOCASA_PACKS if not (ROBOCASA_DIR / pack).is_dir()]
    if pending:
        print(
            f"[robocasa] fetching {', '.join(pending)} from utexas.box.com "
            "(2.8 GB, unpacks to 7.5 GB)"
        )
    for pack, pack_id in ROBOCASA_PACKS:
        dest = ROBOCASA_DIR / pack
        if dest.is_dir():
            print(f"[skip] robocasa {pack} (already present)")
            continue
        print(f"[robocasa] {pack}")
        zip_path = ROBOCASA_DIR / f"{pack}.zip"
        download_url(ROBOCASA_URL.format(pack_id=pack_id), zip_path, require_size=False)
        staging = dest.with_name(dest.name + ".part")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            check_members(staging, zf.namelist())
            zf.extractall(staging)
        staging.rename(dest)
        zip_path.unlink(missing_ok=True)


def resolve_sim_task(name):
    """Map a task name, alias, or prompt onto an abc_sim scene-task name."""
    if name.startswith(VLA_SIM_CHECKPOINT_PREFIX):
        name = name.removeprefix(VLA_SIM_CHECKPOINT_PREFIX)
    if name in SIM_TASK_PACKAGES:
        return name
    try:
        from abc_sim.task_specs import get_task_spec
    except ImportError as e:
        raise SystemExit(
            f"error: unknown --sim-task {name!r} (abc_sim is not importable: {e})"
        ) from None
    try:
        return get_task_spec(name).env_task
    except KeyError:
        raise SystemExit(f"error: unknown --sim-task {name!r}") from None


def sim_task_packages(tasks):
    wanted = set(SIM_BASE_PACKAGES)
    for task in tasks:
        scene = resolve_sim_task(task)
        extra = SIM_TASK_PACKAGES.get(scene)
        if extra is None:
            mapped = ", ".join(sorted(SIM_TASK_PACKAGES))
            raise SystemExit(
                f"error: no asset map for task {task!r} (scene {scene!r}).\n"
                f"       mapped scenes: {mapped}\n"
                f"       use --sim to install every package instead."
            )
        wanted.update(extra)
    return wanted


def fetch_sim_data_manifest(cache, *, force=False):
    """Return the sim_224 data manifest, caching it under the cache dir.

    ``--sim-force`` refetches it; otherwise the cached copy is reused, so a run that
    only needs one more task does not re-read the whole release description.
    """
    path = cache / DATA_MANIFEST_CACHE
    if force or not path.exists():
        url = f"{DATA_BASE}/{DATA_MANIFEST_KEY}"
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(
                    f"no sim_224 manifest at <{url}> yet; the release is still uploading"
                ) from None
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = json.loads(path.read_text())
    if manifest.get("format") != DATA_MANIFEST_FORMAT:
        raise RuntimeError(f"unexpected manifest format in {path}: {manifest.get('format')!r}")
    return manifest


def sim_data_scene(name):
    """Map one --sim-data name onto an abc_sim scene."""
    if name in DATASET_TASK_SCENES:
        return DATASET_TASK_SCENES[name]
    try:
        return resolve_sim_task(name)
    except SystemExit:
        raise SystemExit(
            f"error: unknown --sim-data {name!r}; it is neither a dataset task name nor a "
            "sim task name.\n       See --sim-data-list for the published tasks."
        ) from None


def sim_data_scenes(names):
    """The scenes behind a --sim-data request, for the asset half of the install."""
    return [sim_data_scene(name) for name in names]


def resolve_sim_data_tasks(names, known):
    """Expand --sim-data names onto dataset task names, in request order, deduplicated.

    An exact dataset task name is taken as itself; anything else resolves through the
    scene, which for the spelling scene means seven tasks. ``known`` is the set of names
    that count as exact -- the manifest's tasks plus the ones this build knows about, so
    resolution still works against a manifest that is missing tasks still uploading.
    """
    resolved = []
    for name in names:
        if name in known:
            tasks = (name,)
        else:
            scene = sim_data_scene(name)
            tasks = SCENE_DATASET_TASKS.get(scene, ())
            if not tasks:
                published = ", ".join(sorted(SCENE_DATASET_TASKS))
                raise SystemExit(
                    f"error: no sim_224 episode data for --sim-data {name!r} "
                    f"(scene {scene!r}).\n       scenes with data: {published}"
                )
            if len(tasks) > 1:
                print(f"[sim-data] {name} -> {len(tasks)} tasks")
        for task in tasks:
            if task not in resolved:
                resolved.append(task)
    return resolved


def sim_data_installed(cache, task_name, task):
    """True when the marker matches the manifest sha and the split dirs are still there.

    Existence only: a task's episodes run to tens of thousands of files, and stat()ing
    them on every run would cost far more than the check saves. Only the splits the
    manifest says carry episodes are required, so a task with an empty val split does
    not look perpetually uninstalled.
    """
    marker = cache / SIM_DATA_MARKER_DIR / f"{task_name}.sha256"
    if not marker.exists() or marker.read_text().strip() != task["sha256"]:
        return False
    return all(
        (cache / f"{split}_sim").is_dir()
        for split, count in task["episodes"].items()
        if count
    )


def install_sim_data(task_name, task, cache):
    """Fetch, verify and unpack one task's episodes, then mark it installed.

    The tar is deleted whatever happens: it is as large as what it unpacks to, and one
    that failed verification must not be trusted by the next run.
    """
    tar = cache / SIM_DATA_TAR_DIR / f"{task_name}.tar"
    try:
        download_url(task["uri"], tar, expected=task["bytes"])
        verify_sha256(tar, task["sha256"], task_name)
        extract_tar(tar, cache, force_role="sim")
        marker = cache / SIM_DATA_MARKER_DIR / f"{task_name}.sha256"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(task["sha256"] + "\n")
    finally:
        tar.unlink(missing_ok=True)


def print_sim_data_list(manifest):
    """One line per published task: what --sim-data can be given and what it costs."""
    tasks = manifest["tasks"]
    print(f"[sim-data] {len(tasks)} task(s), generated {manifest.get('generated', '?')}")
    for name, task in tasks.items():
        episodes = task["episodes"]
        print(
            f"{name:46s} {_fmt_bytes(task['bytes']):>9s}  "
            f"{episodes['train']:5d} train  {episodes['val']:4d} val  "
            f"{task['frames']:8d} frames  {task['hours']:6.1f} h"
        )


def fetch_ckpt_manifest(cache, *, force=False):
    """Return the finetuned-checkpoint manifest, caching it under the cache dir."""
    path = cache / CKPT_MANIFEST_CACHE
    if force or not path.exists():
        url = f"{DATA_BASE}/{CKPT_MANIFEST_KEY}"
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"no checkpoint manifest at <{url}> yet") from None
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = json.loads(path.read_text())
    if manifest.get("format") != CKPT_MANIFEST_FORMAT:
        raise RuntimeError(f"unexpected manifest format in {path}: {manifest.get('format')!r}")
    return manifest


def print_sim_checkpoint_list(manifest, *, label="sim-checkpoint"):
    """One line per task: published steps (* = recommended), its eval score, prompt."""
    tasks = manifest["tasks"]
    print(f"[{label}] {len(tasks)} task(s), generated {manifest.get('generated', '?')}")
    for name, task in tasks.items():
        rec = str(task["recommended_step"])
        steps = ",".join(
            s + ("*" if s == rec else "") for s in sorted(task["checkpoints"], key=int)
        )
        scores = [
            e for label, e in task["evals"].items()
            if label.split("_")[0] == rec and e.get("num_success") is not None
        ]
        best = (
            "+".join(str(e["num_success"]) for e in scores)
            + f"/{sum(e['num_worlds'] for e in scores)}"
            if scores else "?"
        )
        print(f"{name:46s} steps {steps:24s} best {best:9s} {task.get('generation', '?')}")


# The finetuned_sim bucket prefixes are the dataset task names, except pour's,
# which predates the sim_pouring_beads spelling.
CKPT_PREFIX_BY_DATASET_TASK = {"sim_pouring_beads": "pour"}


def resolve_sim_checkpoint_task(name, tasks):
    """Manifest key for a --sim-checkpoint name: exact, else via the task aliases.

    A ``vla_`` name only ever resolves to a ``vla_`` key, never to the DiT entry."""
    if name in tasks:
        return name
    prefix = VLA_SIM_CHECKPOINT_PREFIX if name.startswith(VLA_SIM_CHECKPOINT_PREFIX) else ""
    try:
        resolved = resolve_sim_data_tasks([name], set(DATASET_TASK_SCENES))
    except SystemExit:
        return None
    for dataset_task in resolved:
        key = prefix + CKPT_PREFIX_BY_DATASET_TASK.get(dataset_task, dataset_task)
        if key in tasks:
            return key
    return None


def prepare_sim_checkpoints(config: PrepareConfig, cache):
    """Download the requested finetuned checkpoints. Returns the number that failed."""
    cache.mkdir(parents=True, exist_ok=True)
    try:
        manifest = fetch_ckpt_manifest(cache, force=config.sim_force)
    except Exception as exc:  # noqa: BLE001 - an unpublished manifest is not a traceback
        print(f"[fail] checkpoint manifest: {sim_failure_reason(exc)}")
        return 1
    if config.sim_checkpoint_list or config.sim_bundle_list:
        label = "sim-bundle" if config.sim_bundle_list else "sim-checkpoint"
        print_sim_checkpoint_list(manifest, label=label)
        return 0

    tasks = manifest["tasks"]
    failed = 0
    requested = dict.fromkeys((*config.sim_checkpoint, *config.sim_bundle))
    for spec in requested:
        name, _, step = spec.partition("@")
        resolved = resolve_sim_checkpoint_task(name, tasks)
        if resolved is None:
            print(f"[fail] {name}: not in the checkpoint manifest; see --sim-checkpoint-list")
            failed += 1
            continue
        if resolved != name:
            print(f"[sim-checkpoint] {name} -> {resolved}")
        name = resolved
        task = tasks[name]
        sim_task = task.get("task", name.removeprefix(VLA_SIM_CHECKPOINT_PREFIX))
        step = step or str(task["recommended_step"])
        ckpt = task["checkpoints"].get(step)
        if ckpt is None:
            steps = ", ".join(sorted(task["checkpoints"], key=int))
            print(f"[fail] {name}@{step}: published steps are {steps}")
            failed += 1
            continue
        uri, size, sha = ckpt["uri"], ckpt["bytes"], ckpt.get("sha256")
        kind = "full training state"
        if not config.sim_checkpoint_full_state and ckpt.get("model_uri"):
            uri, size = ckpt["model_uri"], ckpt["model_bytes"]
            sha, kind = ckpt.get("model_sha256"), "model-only"
        dst = cache / "finetuned_sim" / name / f"{step}.pt"
        if (kind == "model-only" and dst.exists()
                and dst.stat().st_size == ckpt["bytes"]):
            # A previously downloaded full-state file is a superset; keep it.
            print(f"[checkpoint] {name}@{step}: full-state file already present, keeping it")
            continue
        print(f"[checkpoint] {name}@{step} ({_fmt_bytes(size)}, {kind})")
        try:
            download_url(uri, dst, expected=size)
            if sha:
                verify_sha256(dst, sha, f"{name}@{step}")
                print(f"[checkpoint] {name}@{step} sha256 ok")
            else:
                # The manifest omits the hash while a republish is in flight.
                print(f"[warn] {name}@{step}: no sha256 in the manifest; size-verified only")
        except Exception as exc:  # noqa: BLE001 - one bad checkpoint cannot strand the rest
            print(f"[fail] {name}@{step}: {sim_failure_reason(exc)}")
            failed += 1
            continue
        if task.get("prompt"):
            print(
                f"[eval] uv run eval_policy.py --checkpoint {dst} --task {sim_task} "
                f"--prompt '{task['prompt']}' --num-worlds 20"
            )
        print(f"[viz] uv run viz_policy.py --sim.checkpoint {dst} --sim.task {sim_task}")
    return failed


def prepare_sim_data(config: PrepareConfig, cache):
    """Download the requested tasks' episode data. Returns the number that failed."""
    cache.mkdir(parents=True, exist_ok=True)
    try:
        manifest = fetch_sim_data_manifest(cache, force=config.sim_force)
    except Exception as exc:  # noqa: BLE001 - an unpublished release is not a traceback
        reason = sim_failure_reason(exc)
        print(f"[fail] sim_224 manifest: {reason}")
        print_sim_summary(
            [],
            [],
            [("sim_224 manifest", reason)],
            label="sim-data",
            noun="task",
            consequence="no episode data was downloaded",
        )
        return 1

    if config.sim_data_list:
        print_sim_data_list(manifest)
        return 0

    tasks = manifest["tasks"]
    wanted = resolve_sim_data_tasks(config.sim_data, set(tasks) | set(DATASET_TASK_SCENES))
    todo = [
        name
        for name in wanted
        if name in tasks
        and (config.sim_force or not sim_data_installed(cache, name, tasks[name]))
    ]
    print(f"[sim-data] {len(wanted)} task(s) -> {cache}")
    print(
        f"[sim-data] {len(todo)} to download, "
        f"{_fmt_bytes(sum(tasks[name]['bytes'] for name in todo))}"
    )

    installed, skipped, failed = [], [], []
    for name in wanted:
        task = tasks.get(name)
        if task is None:
            # The release uploads smallest-first, so a known task can legitimately be
            # absent from a manifest this run already fetched.
            reason = "not in the published manifest yet"
            print(f"[fail] {name}: {reason}")
            failed.append((name, reason))
            continue
        if not config.sim_force and sim_data_installed(cache, name, task):
            print(f"[skip] {name} (already present)")
            skipped.append(name)
            continue
        episodes = sum(task["episodes"].values())
        print(f"[data] {name} ({_fmt_bytes(task['bytes'])}, {episodes} episodes)")
        try:
            install_sim_data(name, task, cache)
        except Exception as exc:  # noqa: BLE001 - one bad task cannot strand the rest
            reason = sim_failure_reason(exc)
            print(f"[fail] {name}: {reason}")
            failed.append((name, reason))
        else:
            installed.append(name)

    print_sim_summary(
        installed,
        skipped,
        failed,
        label="sim-data",
        noun="task",
        consequence="their episodes are missing from the cache",
    )
    if not failed:
        print("[done] sim episode data ready")
    return len(failed)


def prepare_sim(config: PrepareConfig):
    """Install the requested asset packages. Returns the number that failed."""
    packages = load_sim_manifest()
    if config.sim_list:
        for pkg in packages.values():
            print(
                f"{pkg['name']:22s} {pkg['size_bytes']/1e6:8.1f} MB  "
                f"{pkg['file_count']:5d} files  -> {pkg['extract_dir']}"
            )
        return 0

    wanted = set()
    if config.sim:
        # Everything except the RoboCasa packs, which pull from a third-party host.
        wanted |= set(packages) - {ROBOCASA_PACKAGE}
    if config.sim_task:
        wanted |= sim_task_packages(config.sim_task)
    if config.sim_bundle:
        wanted |= sim_task_packages(config.sim_bundle)
    if config.sim_data:
        # Episode data is only useful next to the meshes it was recorded against, so
        # --sim-data installs its tasks' assets too.
        wanted |= sim_task_packages(sim_data_scenes(config.sim_data))
    for name in config.sim_package:
        if name not in packages:
            raise SystemExit(f"error: unknown --sim-package {name!r}; see --sim-list")
        wanted.add(name)
    if config.sim_robocasa:
        wanted.add(ROBOCASA_PACKAGE)

    source = None
    if config.sim_source is not None:
        source = config.sim_source.expanduser().resolve()
        if not source.is_dir():
            raise SystemExit(f"error: --sim-source is not a directory: {source}")

    order = [pkg for name, pkg in packages.items() if name in wanted and name != ROBOCASA_PACKAGE]
    if ROBOCASA_PACKAGE in wanted:
        order.append(packages[ROBOCASA_PACKAGE])
    todo = [pkg for pkg in order if config.sim_force or not sim_installed(pkg)]
    print(f"[sim] {len(order)} package(s) -> {SIM_MODELS_DIR}")
    print(f"[sim] {len(todo)} to install, {_fmt_bytes(sum(pkg['size_bytes'] for pkg in todo))}")

    installed, skipped, failed = [], [], []
    with tempfile.TemporaryDirectory(prefix="abc-sim-assets-") as work:
        for pkg in order:
            name = pkg["name"]
            if sim_installed(pkg) and not config.sim_force:
                print(f"[skip] {name} (already installed)")
                skipped.append(name)
                continue
            print(f"[asset] {name} ({pkg['size_bytes']/1e6:.1f} MB, {pkg['file_count']} files)")
            try:
                install_sim_package(pkg, source, Path(work))
            except Exception as exc:  # noqa: BLE001 - one bad package cannot strand the rest
                reason = sim_failure_reason(exc)
                print(f"[fail] {name}: {reason}")
                failed.append((name, reason))
            else:
                installed.append(name)

    if ROBOCASA_PACKAGE in wanted:
        try:
            fetch_robocasa_packs()
        except Exception as exc:  # noqa: BLE001 - report it, do not traceback
            reason = sim_failure_reason(exc)
            print(f"[fail] robocasa object packs: {reason}")
            failed.append(("robocasa object packs", reason))
    elif config.sim:
        print(
            "[note] skipped assets_robocasa: only the inhand_transfer scene needs it, and its "
            "object packs are a 2.8 GB download from utexas.box.com. Pass --sim-robocasa for them."
        )
    print_sim_summary(installed, skipped, failed)
    if not failed:
        print("[done] sim assets ready")
    return len(failed)


def main(config: PrepareConfig):
    # Keep status lines interleaved with the stderr progress bars when output is a pipe.
    sys.stdout.reconfigure(line_buffering=True)
    pretrained_requested = config.pretrained or config.vla_pretrained
    assets_requested = bool(
        config.sim
        or config.sim_task
        or config.sim_bundle
        or config.sim_package
        or config.sim_robocasa
        or config.sim_list
    )
    data_requested = bool(config.sim_data or config.sim_data_list)
    ckpt_requested = bool(
        config.sim_checkpoint
        or config.sim_bundle
        or config.sim_checkpoint_list
        or config.sim_bundle_list
    )
    sim_requested = assets_requested or data_requested or ckpt_requested
    if not (sim_requested or pretrained_requested) and (
        config.sim_source is not None or config.sim_force
    ):
        raise SystemExit(
            "error: --sim-source/--sim-force only apply with --sim, --sim-task, "
            "--sim-bundle, --sim-package or --sim-data"
        )
    cache = config.cache.expanduser().resolve()
    if config.dagger_checkpoint:
        fetch_dagger_checkpoints(cache, config.dagger_checkpoint)
        if not (sim_requested or config.full or config.checkpoint or pretrained_requested):
            return
    sim_failures = 0
    if sim_requested:
        # Assets first: --sim-data installs its tasks' packages as well, and the meshes
        # are what makes the episodes replayable.
        if assets_requested or config.sim_data:
            sim_failures = prepare_sim(config)
        if data_requested:
            sim_failures += prepare_sim_data(config, cache)
        if ckpt_requested:
            sim_failures += prepare_sim_checkpoints(config, cache)
        # Sim asset flags on their own do not imply the bottles dataset; ask for that
        # explicitly. --sim-data is itself a dataset download, so it exits here too.
        if not (config.full or config.checkpoint or pretrained_requested):
            raise SystemExit(1 if sim_failures else 0)

    cache.mkdir(parents=True, exist_ok=True)
    print(f"[cache] {cache}")

    print("[small] norm_stats.json")
    for key, name in SMALL_FILES:
        download(key, cache / name)

    if config.full or config.checkpoint or not pretrained_requested:
        tars = FULL_TARS if config.full else [PREVIEW_TAR]
        fetch_tars(cache, tars, skip_extract=config.skip_extract)

    if config.checkpoint:
        print(f"[checkpoint] {CHECKPOINT_KEY}")
        download(CHECKPOINT_KEY, cache / CHECKPOINT_DST)

    if config.pretrained:
        fetch_pretrained(cache, config)

    if config.vla_pretrained:
        fetch_vla_pretrained(cache, config)

    print("\n[done] cache layout:")
    for entry in sorted(cache.iterdir()):
        if entry.is_dir():
            n = sum(1 for _ in entry.iterdir())
            print(f"  {entry.name}/  ({n} entries)")
        else:
            print(f"  {entry.name}  ({entry.stat().st_size/1e6:.1f} MB)")

    if sim_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(PrepareConfig))
