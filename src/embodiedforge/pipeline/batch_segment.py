"""Batch segmentation runner — processes an entire dataset directory.

Phase 2 of the data-centric segmentation workflow:
1. Scans dataset directory for all episodes
2. Runs rule-based segmenter on each episode
3. Writes per-episode artifacts (candidates, draft, refine stub)
4. Produces a dataset-level index with status and summary stats

Failures in individual episodes are caught and logged — they never kill the batch.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from embodiedforge.sampling.schemas import DatasetSegmentIndex, EpisodeSegmentResult


def run_batch_segment(
    dataset_root: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    max_episodes: int = 0,
    episode_ids: list[str] | None = None,
) -> DatasetSegmentIndex:
    """Run rule-based segmentation on every episode in a dataset directory.

    Args:
        dataset_root: Root directory containing episode subdirectories (standard or xlsx).
        output_dir: Root output directory — per-episode artifacts go into per-episode subdirs.
        config: Inline config dict (takes precedence over config_path).
        config_path: Path to a pipeline config YAML.
        max_episodes: Limit processing to N episodes (0 = all). Useful for quick checks.
        episode_ids: Only process these specific episode IDs (by directory name).

    Returns:
        DatasetSegmentIndex with per-episode results and summary statistics.
    """
    from embodiedforge.sampling.scanner import scan_dataset
    from embodiedforge.pipeline.segment_demo import run_segment_demo
    from embodiedforge.pipeline.run_all import load_config

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    # Resolve config
    if config is None:
        if config_path is None:
            config_path = Path("configs/demo.yaml")
        config = load_config(config_path)
    config_path_str = str(config_path) if config_path else ""

    # Discover episodes
    console = Console()
    console.print(f"[bold]Batch segmentation[/]")
    console.print(f"  Dataset: {dataset_root}")
    console.print(f"  Output:  {output_dir}")

    episodes = scan_dataset(dataset_root)
    if episode_ids:
        episodes = [ep for ep in episodes if ep.episode_id in set(episode_ids)]
        console.print(f"  Filtering to {len(episodes)} specified episode(s)")
    if max_episodes > 0 and len(episodes) > max_episodes:
        episodes = episodes[:max_episodes]
        console.print(f"  Limiting to first {max_episodes} episode(s)")

    console.print(f"  Processing {len(episodes)} episode(s)...\n")

    # Progress bar
    results: list[EpisodeSegmentResult] = []
    success_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running segmenter...", total=len(episodes))

        for ep_summary in episodes:
            ep_dir = Path(ep_summary.source_path)
            ep_output = output_dir / ep_summary.episode_id
            progress.update(task, description=f"[cyan]{ep_summary.episode_id}")
            t0 = time.perf_counter()

            try:
                seg_result = run_segment_demo(
                    episode_dir=ep_dir,
                    output_dir=ep_output,
                    config=config,
                )
                elapsed = round(time.perf_counter() - t0, 2)

                results.append(
                    EpisodeSegmentResult(
                        episode_id=ep_summary.episode_id,
                        source_path=str(ep_dir),
                        status="success",
                        stage_count=seg_result.get("stage_count", 0),
                        boundary_count=_count_boundaries(ep_output),
                        num_frames=ep_summary.num_frames,
                        method=seg_result.get("method", "unknown"),
                        elapsed_seconds=elapsed,
                        artifacts_dir=str(ep_output / "segments"),
                    )
                )
                success_count += 1

            except Exception as exc:
                elapsed = round(time.perf_counter() - t0 if 't0' in dir() else 0, 2)
                error_msg = f"{type(exc).__name__}: {exc}"
                results.append(
                    EpisodeSegmentResult(
                        episode_id=ep_summary.episode_id,
                        source_path=str(ep_dir),
                        status="error",
                        error_message=error_msg,
                        elapsed_seconds=elapsed,
                        artifacts_dir=str(ep_output / "segments"),
                    )
                )
                error_count += 1
                # Log but continue
                progress.console.print(f"  [red]✗ {ep_summary.episode_id}: {error_msg[:120]}[/]")

            progress.update(task, advance=1)

    # Build index
    index = DatasetSegmentIndex(
        dataset_root=str(dataset_root),
        config_path=config_path_str,
        total_episodes=len(results),
        success_count=success_count,
        error_count=error_count,
        episodes=results,
        summary=_build_summary(results),
    )

    # Write index
    index_dir = output_dir / "batch"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "dataset_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index.model_dump_json(indent=2))

    # Print report
    _print_batch_report(console, index, output_dir)

    return index


def _count_boundaries(ep_output: Path) -> int:
    """Count boundary candidates in the written artifact."""
    candidates_path = ep_output / "segments" / "boundary_candidates.json"
    if not candidates_path.exists():
        return 0
    try:
        with open(candidates_path, encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("candidates", []))
    except (json.JSONDecodeError, OSError):
        return 0


def _build_summary(results: list[EpisodeSegmentResult]) -> dict[str, Any]:
    """Compute dataset-level summary statistics."""
    success_results = [r for r in results if r.status == "success"]
    if not success_results:
        return {"total": len(results), "success": 0, "error": len(results)}

    stages_list = [r.stage_count for r in success_results]
    boundaries_list = [r.boundary_count for r in success_results]
    times_list = [r.elapsed_seconds for r in success_results]

    # Collect skill distribution
    skill_counts: dict[str, int] = {}
    for r in success_results:
        for skill in r.skill_labels:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    return {
        "total": len(results),
        "success": len(success_results),
        "error": len(results) - len(success_results),
        "total_stages": sum(stages_list),
        "avg_stages_per_episode": round(sum(stages_list) / len(success_results), 1) if success_results else 0,
        "min_stages": min(stages_list) if stages_list else 0,
        "max_stages": max(stages_list) if stages_list else 0,
        "total_boundaries": sum(boundaries_list),
        "avg_boundaries_per_episode": round(sum(boundaries_list) / len(success_results), 1) if success_results else 0,
        "total_elapsed_seconds": round(sum(times_list), 2),
        "avg_elapsed_per_episode_seconds": round(sum(times_list) / len(success_results), 2) if success_results else 0,
        "skill_distribution": skill_counts,
    }


def _print_batch_report(console: Console, index: DatasetSegmentIndex, output_dir: Path) -> None:
    """Print a formatted batch report table."""
    s = index.summary

    console.print(f"\n[bold green]✓ Batch complete[/]")
    console.print(f"  {s['total']} episode(s) processed: {s['success']} success, {s['error']} error(s)")

    table = Table(title="Per-Episode Results", show_lines=False)
    table.add_column("Episode", style="cyan", width=25)
    table.add_column("Status", width=8)
    table.add_column("Stages", justify="right", width=7)
    table.add_column("Bounds", justify="right", width=7)
    table.add_column("Time", justify="right", width=7)
    table.add_column("Error", style="red", width=30)

    for r in index.episodes:
        status_style = "green" if r.status == "success" else "red"
        table.add_row(
            r.episode_id[:24],
            f"[{status_style}]{r.status}[/]",
            str(r.stage_count),
            str(r.boundary_count),
            f"{r.elapsed_seconds:.1f}s",
            r.error_message[:28] if r.error_message else "",
        )

    console.print(table)

    if s["success"] > 0:
        console.print(f"\n[bold]Summary stats:[/]")
        console.print(f"  Stages:     {s['total_stages']} total, {s['avg_stages_per_episode']} avg, "
                      f"{s['min_stages']}–{s['max_stages']} range")
        console.print(f"  Boundaries: {s['total_boundaries']} total, {s['avg_boundaries_per_episode']} avg")
        console.print(f"  Time:       {s['total_elapsed_seconds']:.1f}s total, "
                      f"{s['avg_elapsed_per_episode_seconds']:.2f}s avg")

    console.print(f"\n[bold]Dataset index:[/] {output_dir}/batch/dataset_index.json")
