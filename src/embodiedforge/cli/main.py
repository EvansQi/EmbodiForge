"""Main CLI entry point.

Usage:
    eforge run-all --episode-dir ./examples/sample_episode --config configs/demo.yaml
    eforge segment --episode-dir ./examples/sample_episode
    eforge annotate-semantic --episode-dir ./examples/sample_episode
    eforge ground --episode-dir ./examples/sample_episode
    eforge segment-mask --episode-dir ./examples/sample_episode
    eforge build-heatmap --episode-dir ./examples/sample_episode
    eforge qc --episode-dir ./examples/sample_episode
    eforge export --episode-dir ./examples/sample_episode
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def cli():
    """EmbodiedForge: Robot episode processing pipeline."""
    pass


@cli.command("run-all")
@click.option("--episode-dir", required=True, type=click.Path(exists=True), help="Path to episode directory")
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path(), help="Pipeline config YAML")
@click.option("--output-dir", default="artifacts", type=click.Path(), help="Output directory")
def run_all_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Run the full pipeline end-to-end."""
    from embodiedforge.pipeline.run_all import run_all

    run_all(episode_dir=episode_dir, config_path=config_path, output_dir=output_dir)


@cli.command("segment")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def segment_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Stage 1: Segment episode into stages."""
    import json

    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.run_all import load_config

    config = load_config(config_path)
    artifacts = Path(output_dir) / "artifacts"
    episode = ingest_episode(episode_dir, config.get("ingest"))
    result = segment_episode(episode, config.get("segment"), artifacts)
    click.echo(f"Found {result.total_stages} stages:")
    for s in result.stages:
        click.echo(f"  [{s.skill_id}] frames {s.start_frame}-{s.end_frame}")


@cli.command("segment-demo")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def segment_demo_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Run the segment-only demo and write draft artifacts."""
    from embodiedforge.pipeline.segment_demo import run_segment_demo

    result = run_segment_demo(episode_dir=episode_dir, output_dir=output_dir, config_path=config_path)
    click.echo(f"Episode: {result['episode_id']}")
    click.echo(f"Draft:   {result['segments_path']}")
    click.echo(f"Bounds:  {result['candidates_path']}")
    click.echo(f"Stages:  {result['stage_count']}")


@cli.command("finalize-segments")
@click.option("--output-dir", required=True, type=click.Path(exists=True))
@click.option("--reviewer", required=True, type=str)
@click.option("--review-notes", default="", type=str)
@click.option("--draft-path", default=None, type=click.Path(exists=True))
def finalize_segments_cmd(output_dir: str, reviewer: str, review_notes: str, draft_path: str | None):
    """Promote a draft segment artifact to segments_final.json."""
    from embodiedforge.pipeline.segment_demo import finalize_segment_demo

    result = finalize_segment_demo(
        output_dir=output_dir,
        reviewer=reviewer,
        review_notes=review_notes,
        draft_path=draft_path,
    )
    click.echo(f"Episode: {result['episode_id']}")
    click.echo(f"Final:   {Path(output_dir) / 'segments' / 'segments_final.json'}")
    click.echo(f"Status:  {result['status']}")


@cli.command("evaluate-segments")
@click.option("--output-dir", required=True, type=click.Path(exists=True))
@click.option("--ground-truth", "ground_truth_path", required=True, type=click.Path(exists=True))
@click.option("--prediction-path", default=None, type=click.Path(exists=True))
def evaluate_segments_cmd(output_dir: str, ground_truth_path: str, prediction_path: str | None):
    """Evaluate a segment artifact against ground truth."""
    import json

    from embodiedforge.pipeline.segment_eval import evaluate_segments

    result = evaluate_segments(
        output_dir=output_dir,
        ground_truth_path=ground_truth_path,
        prediction_path=prediction_path,
    )
    click.echo(json.dumps(result, indent=2))


@cli.command("sample-dataset")
@click.option("--dataset-root", required=True, type=click.Path(exists=True), help="Root directory containing episode subdirectories")
@click.option("--output-dir", default="artifacts", type=click.Path(), help="Output directory for manifest and split files")
@click.option("--target-size", default=30, type=int, help="Target number of episodes for ground truth set")
@click.option("--dimensions", default="task_type,duration_bucket,camera_setup,success", type=str, help="Comma-separated stratification dimensions")
@click.option("--min-per-stratum", default=1, type=int, help="Minimum episodes per stratum")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("--scan-only", is_flag=True, default=False, help="Only scan, don't build split")
def sample_dataset_cmd(
    dataset_root: str,
    output_dir: str,
    target_size: int,
    dimensions: str,
    min_per_stratum: int,
    seed: int,
    scan_only: bool,
):
    """Phase 1: Scan dataset and build stratified ground truth split.

    Scans a dataset directory tree, extracts episode metadata, performs
    stratified sampling, and outputs:

        artifacts/sampling/sampling_manifest.json  — all episodes + strata + rationale
        artifacts/sampling/ground_truth_split.json — selected episodes for annotation
    """
    import json

    from embodiedforge.sampling.scanner import scan_dataset, scan_single_episode
    from embodiedforge.sampling.sampler import build_sampling_manifest, build_ground_truth_split

    from rich.console import Console
    from rich.table import Table

    console = Console()

    dim_list = [d.strip() for d in dimensions.split(",") if d.strip()]

    # Scan
    console.print(f"[bold]Scanning dataset:[/] {dataset_root}")
    console.print(f"Dimensions: {dim_list}")
    episodes = scan_dataset(dataset_root)

    console.print(f"\n[bold]Found {len(episodes)} episode(s)[/]")
    complete = sum(1 for ep in episodes if ep.is_complete)
    console.print(f"  Complete: {complete}")
    console.print(f"  With issues: {len(episodes) - complete}")

    # Summary table
    table = Table(title="Episode Summary")
    table.add_column("Episode ID", style="cyan")
    table.add_column("Task Type", style="green")
    table.add_column("Frames")
    table.add_column("Cameras")
    table.add_column("Success")
    table.add_column("Complete")

    for ep in episodes[:20]:  # Show first 20
        table.add_row(
            ep.episode_id,
            ep.task_type,
            str(ep.num_frames),
            ep.camera_setup,
            "✓" if ep.success else ("✗" if ep.success is False else "?"),
            "✓" if ep.is_complete else "✗",
        )
    if len(episodes) > 20:
        table.add_row("...", "...", "...", "...", "...", f"+{len(episodes) - 20} more")

    console.print(table)

    if scan_only:
        # Just save the manifest metadata
        output_path = Path(output_dir) / "sampling"
        output_path.mkdir(parents=True, exist_ok=True)
        from embodiedforge.sampling.schemas import SamplingManifest

        manifest = SamplingManifest(
            dataset_root=dataset_root,
            total_episodes=len(episodes),
            complete_episodes=complete,
            sampling_dimensions=dim_list,
            episodes=episodes,
        )
        with open(output_path / "sampling_manifest.json", "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        console.print(f"\n[bold green]Scan-only manifest written to:[/] {output_path / 'sampling_manifest.json'}")
        return

    # Build manifest with stratified sampling
    manifest = build_sampling_manifest(
        episodes=episodes,
        dataset_root=dataset_root,
        target_sample_size=target_size,
        dimensions=dim_list,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    # Build ground truth split
    output_path = Path(output_dir) / "sampling"
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "sampling_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    split = build_ground_truth_split(
        manifest,
        manifest_path=str(manifest_path),
    )
    split_path = output_path / "ground_truth_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        f.write(split.model_dump_json(indent=2))

    console.print(f"\n[bold green]Sampling complete![/]")
    console.print(f"  Selected: {len(manifest.selected_ids)} / {len(episodes)} episodes")
    console.print(f"  Manifest: {manifest_path}")
    console.print(f"  Split:    {split_path}")
    console.print(f"\n[bold]Sampling Rationale:[/]")
    console.print(manifest.sampling_rationale)
    console.print(f"\n[bold]Next step:[/] Annotate selected episodes with:")
    console.print("  eforge segment-demo --episode-dir <episode_dir>")


@cli.command("scan-episode")
@click.option("--episode-dir", required=True, type=click.Path(exists=True), help="Path to a single episode directory")
def scan_episode_cmd(episode_dir: str):
    """Scan a single episode and print its summary."""
    import json

    from embodiedforge.sampling.scanner import scan_single_episode

    summary = scan_single_episode(episode_dir)
    click.echo(summary.model_dump_json(indent=2))


@cli.command("batch-segment")
@click.option("--dataset-root", required=True, type=click.Path(exists=True), help="Root directory containing episode subdirectories")
@click.option("--output-dir", default="artifacts", type=click.Path(), help="Root output directory for per-episode results")
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path(), help="Pipeline config YAML")
@click.option("--max-episodes", default=0, type=int, help="Limit to N episodes (0=all)")
@click.option("--episode-ids", default=None, type=str, help="Comma-separated episode IDs to process (overrides max-episodes)")
def batch_segment_cmd(
    dataset_root: str,
    output_dir: str,
    config_path: str,
    max_episodes: int,
    episode_ids: str | None,
):
    """Phase 2: Run segmentation on every episode in a dataset directory.

    Discovers all episodes (standard + xlsx formats), runs the rule-based
    segmenter on each, and writes:

        <output_dir>/<episode_id>/segments/
            boundary_candidates.json
            segments_draft.json
            boundary_refine_vlm.json

    A dataset-level index is written to <output_dir>/batch/dataset_index.json.
    Failures in individual episodes are logged but never stop the batch.
    """
    id_list = [e.strip() for e in episode_ids.split(",") if e.strip()] if episode_ids else None

    from embodiedforge.pipeline.batch_segment import run_batch_segment

    run_batch_segment(
        dataset_root=dataset_root,
        output_dir=output_dir,
        config_path=config_path,
        max_episodes=max_episodes,
        episode_ids=id_list,
    )


@cli.command("annotate-semantic")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def annotate_semantic_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Stage 2: Semantic annotation of keyframes."""
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.annotate_semantic import annotate_semantic
    from embodiedforge.pipeline.run_all import load_config

    config = load_config(config_path)
    artifacts = Path(output_dir) / "artifacts"
    episode = ingest_episode(episode_dir, config.get("ingest"))
    segments = segment_episode(episode, config.get("segment"), artifacts)
    result = annotate_semantic(episode, segments, config.get("semantic"), artifacts)
    click.echo(f"Annotated {len(result.annotations)} keyframes:")
    for ann in result.annotations:
        click.echo(f"  f{ann.frame_idx}: {ann.target_object} / {ann.target_part}")


@cli.command("ground")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def ground_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Stage 3: Visual grounding of objects."""
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.annotate_semantic import annotate_semantic
    from embodiedforge.pipeline.ground import ground_objects
    from embodiedforge.pipeline.run_all import load_config

    config = load_config(config_path)
    artifacts = Path(output_dir) / "artifacts"
    episode = ingest_episode(episode_dir, config.get("ingest"))
    segments = segment_episode(episode, config.get("segment"), artifacts)
    semantic = annotate_semantic(episode, segments, config.get("semantic"), artifacts)
    results = ground_objects(episode, semantic, config.get("grounding"), artifacts)
    click.echo(f"Grounded {len(results)} frames:")
    for idx, boxes in results.items():
        obj = boxes.get("object_bbox")
        click.echo(f"  f{idx}: obj_bbox={obj}")


@cli.command("segment-mask")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def segment_mask_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Stage 4: Generate segmentation masks."""
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.annotate_semantic import annotate_semantic
    from embodiedforge.pipeline.ground import ground_objects
    from embodiedforge.pipeline.segment_mask import segment_masks
    from embodiedforge.pipeline.run_all import load_config

    config = load_config(config_path)
    artifacts = Path(output_dir) / "artifacts"
    episode = ingest_episode(episode_dir, config.get("ingest"))
    segments = segment_episode(episode, config.get("segment"), artifacts)
    semantic = annotate_semantic(episode, segments, config.get("semantic"), artifacts)
    grounding = ground_objects(episode, semantic, config.get("grounding"), artifacts)
    results = segment_masks(episode, grounding, config.get("segmentation"), artifacts)
    click.echo(f"Generated {len(results)} masks:")
    for idx, mr in results.items():
        click.echo(f"  f{idx}: area={mr.mask_area}, path={mr.mask_path}")


@cli.command("build-heatmap")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def build_heatmap_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Stage 5: Build affordance heatmaps."""
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.annotate_semantic import annotate_semantic
    from embodiedforge.pipeline.ground import ground_objects
    from embodiedforge.pipeline.segment_mask import segment_masks
    from embodiedforge.pipeline.build_heatmap import build_heatmaps
    from embodiedforge.pipeline.run_all import load_config

    config = load_config(config_path)
    artifacts = Path(output_dir) / "artifacts"
    episode = ingest_episode(episode_dir, config.get("ingest"))
    segments = segment_episode(episode, config.get("segment"), artifacts)
    semantic = annotate_semantic(episode, segments, config.get("semantic"), artifacts)
    grounding = ground_objects(episode, semantic, config.get("grounding"), artifacts)
    masks = segment_masks(episode, grounding, config.get("segmentation"), artifacts)
    results = build_heatmaps(episode, semantic, masks, grounding, config.get("affordance"), artifacts)
    click.echo(f"Built {len(results)} heatmaps:")
    for idx, ar in results.items():
        click.echo(f"  f{idx}: max={ar.heatmap_max:.2f}, path={ar.heatmap_path}")


@cli.command("qc")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def qc_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Stage 7: Run quality control checks."""
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.annotate_semantic import annotate_semantic
    from embodiedforge.pipeline.ground import ground_objects
    from embodiedforge.pipeline.segment_mask import segment_masks
    from embodiedforge.pipeline.build_heatmap import build_heatmaps
    from embodiedforge.qc.checks import run_qc
    from embodiedforge.pipeline.run_all import load_config

    config = load_config(config_path)
    artifacts = Path(output_dir) / "artifacts"
    episode = ingest_episode(episode_dir, config.get("ingest"))
    segments = segment_episode(episode, config.get("segment"), artifacts)
    semantic = annotate_semantic(episode, segments, config.get("semantic"), artifacts)
    grounding = ground_objects(episode, semantic, config.get("grounding"), artifacts)
    masks = segment_masks(episode, grounding, config.get("segmentation"), artifacts)
    affordance = build_heatmaps(episode, semantic, masks, grounding, config.get("affordance"), artifacts)
    qc = run_qc(episode.meta.episode_id, masks, affordance, segments, config.get("qc"))
    click.echo(f"QC Score: {qc.score:.1%}")
    for check in qc.checks:
        status = click.style("PASS", fg="green") if check.passed else click.style("FAIL", fg="red")
        click.echo(f"  [{status}] {check.name}: {check.message}")


@cli.command("build-rl-signals")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def build_rl_signals_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Build reward, success, and binary classifier targets for RL experiments."""
    from embodiedforge.pipeline.annotate_3d import annotate_3d
    from embodiedforge.pipeline.annotate_semantic import annotate_semantic
    from embodiedforge.pipeline.build_heatmap import build_heatmaps
    from embodiedforge.pipeline.ground import ground_objects
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.segment_mask import segment_masks
    from embodiedforge.pipeline.run_all import load_config
    from embodiedforge.qc.checks import run_qc
    from embodiedforge.rl.signals import build_rl_signals

    config = load_config(config_path)
    artifacts = Path(output_dir) / "artifacts"
    episode = ingest_episode(episode_dir, config.get("ingest"))
    segments = segment_episode(episode, config.get("segment"), artifacts)
    semantic = annotate_semantic(episode, segments, config.get("semantic"), artifacts)
    grounding = ground_objects(episode, semantic, config.get("grounding"), artifacts)
    masks = segment_masks(episode, grounding, config.get("segmentation"), artifacts)
    affordance = build_heatmaps(episode, semantic, masks, grounding, config.get("affordance"), artifacts)
    geometry = annotate_3d(episode, affordance, config.get("depth"), artifacts)
    qc = run_qc(episode.meta.episode_id, masks, affordance, segments, config.get("qc"))
    signals, summary = build_rl_signals(
        episode, segments, semantic, grounding, masks, affordance, geometry, qc, config.get("rl")
    )

    click.echo(
        f"Built {len(signals)} RL signals: success={summary.success}, "
        f"dense_return={summary.total_dense_reward:.2f}, sparse_return={summary.total_sparse_reward:.2f}"
    )
    for frame_idx, signal in sorted(signals.items()):
        click.echo(
            f"  f{frame_idx}: reward={signal.dense_reward:.2f}, "
            f"success={signal.success}, subgoal={signal.subgoal_completed}"
        )


@cli.command("export")
@click.option("--episode-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="configs/demo.yaml", type=click.Path())
@click.option("--output-dir", default="artifacts", type=click.Path())
def export_cmd(episode_dir: str, config_path: str, output_dir: str):
    """Stage 9: Export training sample."""
    click.echo("Use 'run-all' to export. Individual export requires full pipeline context.")


@cli.command("generate-sample-data")
@click.option("--output-dir", default="examples/sample_episode", type=click.Path())
def generate_sample_data_cmd(output_dir: str):
    """Generate synthetic sample episode data for testing."""
    from embodiedforge.cli.generate_data import generate_sample_episode

    generate_sample_episode(output_dir)
    click.echo(f"Sample episode generated at {output_dir}")


if __name__ == "__main__":
    cli()
