"""Run-all orchestrator — executes the full pipeline end-to-end.

This is the single entry point that chains all stages:
ingest -> segment -> annotate_semantic -> ground -> mask -> heatmap -> 3d -> qc -> viz -> export
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from embodiedforge.export.exporter import export_sample
from embodiedforge.pipeline.annotate_3d import annotate_3d
from embodiedforge.pipeline.annotate_semantic import annotate_semantic
from embodiedforge.pipeline.build_heatmap import build_heatmaps
from embodiedforge.pipeline.ground import ground_objects
from embodiedforge.pipeline.ingest import ingest_episode
from embodiedforge.pipeline.segment import segment_episode
from embodiedforge.pipeline.segment_mask import segment_masks
from embodiedforge.qc.checks import run_qc
from embodiedforge.schemas.sample import TrainingSample
from embodiedforge.viz.overlay import generate_overlays


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_all(
    episode_dir: str | Path,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path = "artifacts",
) -> TrainingSample:
    """Run the full pipeline on a single episode.

    Args:
        episode_dir: Path to the episode directory.
        config: Pipeline configuration dict. If None, loads from config_path.
        config_path: Path to YAML config file. Used if config is None.
        output_dir: Base output directory for artifacts and exports.

    Returns:
        The final TrainingSample.
    """
    if config is None:
        if config_path is None:
            config_path = Path("configs/demo.yaml")
        config = load_config(config_path)

    episode_dir = Path(episode_dir)
    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "artifacts"
    export_dir = output_dir / "export"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    stage_times: dict[str, float] = {}
    t0 = time.time()

    # Stage 0: Ingest
    print("[Stage 0] Ingesting episode...")
    t = time.time()
    episode = ingest_episode(episode_dir, config.get("ingest"))
    stage_times["ingest"] = time.time() - t
    print(f"  -> {len(episode)} frames, task: '{episode.meta.task_instruction}'")

    # Stage 1: Segment
    print("[Stage 1] Segmenting into stages...")
    t = time.time()
    segments = segment_episode(episode, config.get("segment"), artifacts_dir)
    stage_times["segment"] = time.time() - t
    print(f"  -> {segments.total_stages} stages found")
    for s in segments.stages:
        print(f"     [{s.skill_id}] frames {s.start_frame}-{s.end_frame}, keyframes: {s.keyframe_indices}")

    # Stage 2: Semantic annotation
    print("[Stage 2] Annotating semantics...")
    t = time.time()
    semantic = annotate_semantic(episode, segments, config.get("semantic"), artifacts_dir)
    stage_times["semantic"] = time.time() - t
    print(f"  -> {len(semantic.annotations)} keyframes annotated")
    for ann in semantic.annotations:
        print(f"     f{ann.frame_idx}: {ann.target_object} / {ann.target_part} -> {ann.affordance_query}")

    # Stage 3: Grounding
    print("[Stage 3] Grounding objects...")
    t = time.time()
    grounding_results = ground_objects(episode, semantic, config.get("grounding"), artifacts_dir)
    stage_times["grounding"] = time.time() - t
    print(f"  -> {len(grounding_results)} frames grounded")

    # Stage 4: Mask generation
    print("[Stage 4] Generating masks...")
    t = time.time()
    mask_results = segment_masks(episode, grounding_results, config.get("segmentation"), artifacts_dir)
    stage_times["mask"] = time.time() - t
    print(f"  -> {len(mask_results)} masks generated")

    # Stage 5: Affordance heatmap
    print("[Stage 5] Building affordance heatmaps...")
    t = time.time()
    affordance_results = build_heatmaps(
        episode, semantic, mask_results, grounding_results, config.get("affordance"), artifacts_dir
    )
    stage_times["heatmap"] = time.time() - t
    print(f"  -> {len(affordance_results)} heatmaps built")

    # Stage 6: 3D geometry
    print("[Stage 6] Annotating 3D geometry...")
    t = time.time()
    geometry_results = annotate_3d(episode, affordance_results, config.get("depth"), artifacts_dir)
    stage_times["3d"] = time.time() - t
    print(f"  -> {len(geometry_results)} frames with 3D keypoints")

    # Stage 7: QC
    print("[Stage 7] Running quality checks...")
    t = time.time()
    qc = run_qc(episode.meta.episode_id, mask_results, affordance_results, segments, config.get("qc"))
    stage_times["qc"] = time.time() - t
    print(f"  -> QC score: {qc.score:.1%} ({sum(1 for c in qc.checks if c.passed)}/{len(qc.checks)} passed)")
    for check in qc.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"     [{status}] {check.name}: {check.message}")

    # Stage 8: Visualization
    print("[Stage 8] Generating visualizations...")
    t = time.time()
    viz_paths = generate_overlays(episode_dir, mask_results, affordance_results, grounding_results, artifacts_dir)
    stage_times["viz"] = time.time() - t
    print(f"  -> {len(viz_paths)} overlay images generated")

    # Stage 9: Export
    print("[Stage 9] Exporting training sample...")
    t = time.time()
    sample = export_sample(
        episode, segments, semantic, grounding_results,
        mask_results, affordance_results, geometry_results,
        qc, export_dir,
    )
    stage_times["export"] = time.time() - t
    print(f"  -> Exported to {export_dir / episode.meta.episode_id / 'sample.json'}")

    total_time = time.time() - t0
    print(f"\nPipeline complete in {total_time:.1f}s")
    print("Stage times:")
    for name, dt in stage_times.items():
        print(f"  {name}: {dt:.2f}s")

    return sample
