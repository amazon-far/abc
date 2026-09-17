"""Export a recorded dataset episode (sim cache format) to an animated USD for Blender.

    python -m abc_sim.rendering.blender.export_episode cache/train_sim/episode_<id> --out shots/count_01 --start 0 --end 600

An episode directory holds scene_assembled.xml, scene_qpos.npy (one row per 30 Hz step) and
episode_metadata.json.  The scene is built from that XML with asset paths resolved against
abc_sim/models, every requested qpos row is loaded and forwarded, and the official mujoco.usd
exporter writes <out>/frames/frame_<n>.usdc plus <out>/geoms.json (see export_usd.py).
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco.usd import exporter as usd_exporter
from pxr import UsdGeom

from abc_sim.rendering.blender.export_usd import geom_records

MODELS = Path(__file__).resolve().parents[2] / "models"


def load_model(xml_path):
    """Assembled scenes reference assets relative to abc_sim/models or its assets/ folder, inconsistently;
    resolve every file attribute against both so the XML loads from anywhere."""
    root = ET.parse(xml_path).getroot()
    for node in root.iter():
        rel = node.get("file")
        if rel and not Path(rel).is_absolute():
            for base in (MODELS / "assets", MODELS):
                if (base / rel).exists():
                    node.set("file", str(base / rel))
                    break
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(MODELS / "assets"))
    compiler.set("texturedir", str(MODELS))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, help="exclusive; default all steps")
    ap.add_argument("--cameras", default="top,left,right")
    ap.add_argument("--active-wrist", default="right", choices=["left", "right"])
    ap.add_argument("--third-camera", default="front")
    args = ap.parse_args()

    episode = Path(args.episode)
    meta_in = json.load(open(episode / "episode_metadata.json"))
    qpos = np.load(episode / "scene_qpos.npy")
    model = load_model(episode / meta_in.get("scene_xml_file", "scene_assembled.xml"))
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"scene_qpos has {qpos.shape[1]} dofs, the rebuilt scene has {model.nq}")
    data = mujoco.MjData(model)
    end = args.end if args.end is not None else qpos.shape[0]
    fps = int(meta_in.get("fps", 30))

    out = Path(args.out).resolve()
    exporter = usd_exporter.USDExporter(
        model=model, output_directory=out.name, output_directory_root=str(out.parent),
        camera_names=args.cameras.split(","), verbose=False,
    )
    for i in range(args.start, end):
        data.qpos[:] = qpos[i]
        data.time = i / fps
        mujoco.mj_forward(model, data)
        exporter.update_scene(data)
    UsdGeom.SetStageMetersPerUnit(exporter.stage, 1.0)
    exporter.stage.SetTimeCodesPerSecond(fps)
    exporter.stage.SetFramesPerSecond(fps)
    exporter.save_scene(filetype="usdc")

    meta = {
        # Sequence tasks carry a prompt_timeline instead of one instruction.
        "name": out.name, "task": meta_in.get("task_name"),
        "prompt": meta_in.get("instruction") or (meta_in.get("prompt_timeline") or [{}])[0].get("prompt"),
        "episode_id": meta_in.get("episode_id", episode.name), "source": "dataset episode",
        "start_frame": args.start, "end_frame": end, "frames": end - args.start, "fps": fps,
        "active_wrist": args.active_wrist, "third_camera": args.third_camera,
        "usd": f"frames/frame_{exporter.frame_count}.usdc",
        "geoms": geom_records(model, exporter),
    }
    (out / "geoms.json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: {end - args.start} states, {len(meta['geoms'])} geoms, prompt {meta['prompt']!r}")


if __name__ == "__main__":
    main()
