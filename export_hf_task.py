# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-protobuf-support", "tyro"]
# ///
"""Export one ABC-130k Hugging Face task to the training format."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro


HF, REPO = "https://huggingface.co", "XDOF/ABC-130k"
CACHE, UA = Path(os.environ.get("ABC_CACHE", "/tmp/abc_minimal_cache")), "abc-minimal-hf-task/1.0"


@dataclass
class Config:
    """Download all MCAPs for one HF task, then run export_mcap.py."""

    task: Annotated[str, tyro.conf.arg(help="HF task folder, e.g. organize_the_condiment_bottles.")]
    split: Annotated[Literal["train", "val", "all"], tyro.conf.arg(help="Split to download.")] = "all"
    repo_id: Annotated[str, tyro.conf.arg(help="HF dataset repo id.")] = REPO
    revision: Annotated[str, tyro.conf.arg(help="HF revision.")] = "main"
    hf_token: Annotated[
        str | None,
        tyro.conf.arg(help="HF token; otherwise use HF_TOKEN or HUGGING_FACE_HUB_TOKEN."),
    ] = None
    cache: Annotated[Path, tyro.conf.arg(help="Cache root.")] = CACHE
    workers: Annotated[int, tyro.conf.arg(help="Workers passed to export_mcap.py.")] = 4
    max_episodes: Annotated[int | None, tyro.conf.arg(help="Optional per-split cap for smoke tests.")] = None
    dry_run: Annotated[bool, tyro.conf.arg(help="List only; do not download or convert.")] = False
    keep_mcaps: Annotated[bool, tyro.conf.arg(help="Keep staged raw MCAPs after conversion.")] = False


def token(cfg: Config) -> str | None:
    vals = [cfg.hf_token, os.getenv("HF_TOKEN"), os.getenv("HUGGING_FACE_HUB_TOKEN")]
    return next((v.strip() for v in vals if v and v.strip()), None)


def headers(tok: str | None) -> dict[str, str]:
    out = {"User-Agent": UA}
    if tok:
        out["Authorization"] = f"Bearer {tok}"
    return out


def quote(path: str) -> str:
    return urllib.parse.quote(path.strip("/"), safe="/")


def next_page(link: str | None) -> str | None:
    if not link:
        return None
    for part in link.split(","):
        if 'rel="next"' in part:
            return part[part.find("<") + 1 : part.find(">")]
    return None


def open_hf(url: str, tok: str | None):
    req = urllib.request.Request(url, headers=headers(tok))
    try:
        return urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                "HF access denied. Accept XDOF/ABC-130k access and set HF_TOKEN "
                "or pass --hf-token."
            ) from e
        if e.code == 404:
            raise RuntimeError(f"HF path not found: {url}") from e
        raise


def read_json(url: str, tok: str | None):
    with open_hf(url, tok) as r:
        return json.loads(r.read()), r.headers.get("Link")


def list_mcaps(cfg: Config, split: str, tok: str | None) -> list[dict]:
    path = f"data/{split}/{cfg.task}"
    url = f"{HF}/api/datasets/{cfg.repo_id}/tree/{cfg.revision}/{quote(path)}?recursive=1&expand=1"
    out = []
    while url:
        payload, link = read_json(url, tok)
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(payload["error"])
        out += [
            {"path": e["path"], "size": int(e.get("size") or 0)}
            for e in payload
            if e.get("type") == "file" and e.get("path", "").endswith("/episode.mcap")
        ]
        url = next_page(link)
    out = sorted(out, key=lambda e: e["path"])
    return out[: cfg.max_episodes] if cfg.max_episodes is not None else out


def fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def split_root(cfg: Config, split: str) -> Path:
    return cfg.cache.expanduser() / "hf_tasks" / cfg.task / split


def local_path(root: Path, item: dict) -> Path:
    p = Path(item["path"]).parts
    return root / p[2] / p[3] / "episode.mcap"


def download(cfg: Config, tok: str | None, item: dict, root: Path) -> None:
    episode = Path(item["path"]).parts[3]
    dst = local_path(root, item)
    if dst.exists() and dst.stat().st_size == item["size"]:
        print(f"[skip] {episode} ({fmt(item['size'])})")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".mcap.part")
    url = f"{HF}/datasets/{cfg.repo_id}/resolve/{cfg.revision}/{quote(item['path'])}"
    start, done = time.monotonic(), 0
    with open_hf(url, tok) as r, open(tmp, "wb") as f:
        while chunk := r.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r[get] {episode} {fmt(done)}/{fmt(item['size'])}", end="")
    print(f"  {time.monotonic() - start:.1f}s")
    if item["size"] and tmp.stat().st_size != item["size"]:
        got = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download for {item['path']}: {got} != {item['size']}")
    tmp.rename(dst)


def write_manifest(cfg: Config, split: str, files: list[dict], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "repo_id": cfg.repo_id,
        "revision": cfg.revision,
        "task": cfg.task,
        "split": split,
        "count": len(files),
        "bytes": sum(f["size"] for f in files),
        "staged_root": str(root),
        "out_dir": str(cfg.cache / f"{split}_real"),
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(data, indent=2))
    print(f"[manifest] {root / 'manifest.json'}")


def convert(cfg: Config, split: str, root: Path) -> None:
    cmd = [sys.executable, "export_mcap.py", str(root), str((cfg.cache / f"{split}_real").expanduser()), str(cfg.workers)]
    print("[convert]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_split(cfg: Config, split: str, tok: str | None) -> None:
    files, root = list_mcaps(cfg, split, tok), split_root(cfg, split)
    print(f"[{split}] {len(files)} episodes, {fmt(sum(f['size'] for f in files))}")
    write_manifest(cfg, split, files, root)
    if cfg.dry_run:
        return
    for item in files:
        download(cfg, tok, item, root)
    convert(cfg, split, root)
    if not cfg.keep_mcaps:
        shutil.rmtree(root)


def main(cfg: Config) -> None:
    tok = token(cfg)
    for split in (("train", "val") if cfg.split == "all" else (cfg.split,)):
        run_split(cfg, split, tok)


if __name__ == "__main__":
    main(tyro.cli(Config))
