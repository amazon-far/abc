# /// script
# requires-python = ">=3.10"
# dependencies = ["tyro"]
# ///
"""Download and unpack the public abc bottles-in-bin dataset.

Fetches preview or full tars plus normalization stats into ABC_CACHE.
"""

import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

DATA_BASE = "https://abc-data.timehorizons.org"
DEFAULT_CACHE = Path(os.environ.get("ABC_CACHE", "/tmp/abc_minimal_cache"))
HTTP_HEADERS = {"User-Agent": "abc-prepare/1.0"}

SMALL_FILES = [
    ("misc/norm_stats.json",                                "norm_stats.json"),
]

PREVIEW_TAR = "dataset/dataset_preview/bottles_in_bin_preview.tar"
FULL_TARS = [
    "dataset/dataset_preview/bottles_in_bin_real.tar",
    "dataset/dataset_preview/bottles_in_bin_sim.tar",
]


@dataclass
class PrepareConfig:
    """Download and unpack the public abc bottles-in-bin dataset."""

    full: Annotated[
        bool,
        tyro.conf.arg(help="Download the 35 GB real + sim tars instead of the preview tar."),
    ] = False
    skip_extract: Annotated[
        bool,
        tyro.conf.arg(help="Leave tars on disk without extracting them."),
    ] = False
    cache: Annotated[
        Path,
        tyro.conf.arg(help=f"Where to put files; defaults to $ABC_CACHE or {DEFAULT_CACHE}."),
    ] = DEFAULT_CACHE


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


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
            return int(resp.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def download(key, dst):
    """Download key to dst with a progress bar. Idempotent on full-size files."""
    url = f"{DATA_BASE}/{key}"
    expected = remote_size(url)
    if expected is None:
        raise RuntimeError(f"object not found: {key}")
    if dst.exists() and dst.stat().st_size == expected:
        print(f"[skip] {dst.name} ({expected/1e6:.1f} MB)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    prefix = f"[get] {dst.name}"
    start = time.monotonic()
    last = 0.0
    done = 0
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
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
    _progress(prefix, done, expected, start, end=True)
    tmp.rename(dst)


def extract_tar(tar_path, cache):
    """Untar bottles_in_bin_*.tar into $ABC_CACHE with a progress bar.

    Tar layout: ``train/<eid>/...``, ``val/<eid>/...``. Each episode's
    metadata.json is read first; episodes whose ``task_name`` starts with
    ``sim_`` go to ``{train,val}_sim/``, the rest to ``{train,val}_real/``.
    """
    print(f"[extract] {tar_path.name}")
    total_size = tar_path.stat().st_size

    # First pass: classify each episode_id by reading metadata.json.
    eid_role = {}  # (split, eid) -> "real" | "sim"
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

    # Second pass: extract files into the routed dir.
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
            role = eid_role.get((parts[0], parts[1]))
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


def main(config: PrepareConfig):
    cache = config.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[cache] {cache}")

    print("[small] norm_stats.json")
    for key, name in SMALL_FILES:
        download(key, cache / name)

    tars = FULL_TARS if config.full else [PREVIEW_TAR]
    fetch_tars(cache, tars, skip_extract=config.skip_extract)

    print("\n[done] cache layout:")
    for entry in sorted(cache.iterdir()):
        if entry.is_dir():
            n = sum(1 for _ in entry.iterdir())
            print(f"  {entry.name}/  ({n} entries)")
        else:
            print(f"  {entry.name}  ({entry.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main(tyro.cli(PrepareConfig))
