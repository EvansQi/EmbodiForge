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
