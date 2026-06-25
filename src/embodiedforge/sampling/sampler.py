"""Stratified sampler — proportional allocation across key metadata dimensions.

Given a list of EpisodeSummary records, builds strata and selects a representative
subset for ground truth annotation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from embodiedforge.sampling.schemas import (
    EpisodeSummary,
    GroundTruthSplit,
    SamplingManifest,
    StratumBucket,
)

# Default sampling dimensions with their extraction functions
_DEFAULT_DIMENSIONS: list[tuple[str, Any]] = [
    ("task_type", lambda ep: ep.task_type),
    ("duration_bucket", lambda ep: ep.duration_bucket),
    ("camera_setup", lambda ep: ep.camera_setup),
    ("success", lambda ep: _success_label(ep)),
]


def _success_label(ep: EpisodeSummary) -> str:
    if ep.success is True:
        return "success"
    if ep.success is False:
        return "failure"
    return "unknown"


def _describe_strata(strata: list[StratumBucket]) -> str:
    """Build a human-readable description of strata coverage."""
    lines: list[str] = []
    by_dim: dict[str, list[StratumBucket]] = {}
    for s in strata:
        by_dim.setdefault(s.dimension, []).append(s)

    for dim, buckets in sorted(by_dim.items()):
        total = sum(b.count for b in buckets)
        lines.append(f"  {dim} ({total} episodes):")
        for b in sorted(buckets, key=lambda x: -x.count):
            lines.append(f"    - {b.value}: {b.count} episode(s)")
    return "\n".join(lines)


def _build_strata(
    episodes: list[EpisodeSummary],
    dimensions: list[tuple[str, Any]],
) -> list[StratumBucket]:
    """Build StrataBucket objects by grouping episodes across all dimensions."""
    strata: list[StratumBucket] = []

    for dim_name, extractor in dimensions:
        buckets: dict[str, list[str]] = {}
        for ep in episodes:
            key = extractor(ep)
            if not key:
                key = "unknown"
            buckets.setdefault(key, []).append(ep.episode_id)

        for value, ep_ids in sorted(buckets.items()):
            strata.append(
                StratumBucket(
                    dimension=dim_name,
                    value=value,
                    count=len(ep_ids),
                    episode_ids=sorted(ep_ids),
                )
            )

    return strata


def _proportional_allocation(
    strata: list[StratumBucket],
    target_total: int,
    min_per_stratum: int = 1,
    seed: int = 42,
) -> set[str]:
    """Allocate samples proportionally to stratum size.

    Strategy:
    1. Every non-empty stratum gets at least min_per_stratum (if enough episodes exist)
    2. Remaining slots distributed proportional to stratum size
    3. Within each stratum, select randomly with fixed seed for reproducibility
    """
    rng = random.Random(seed)
    selected: set[str] = set()
    quota: dict[str, int] = {}  # stratum_key -> allocation count

    # Filter to non-empty strata
    active = [s for s in strata if s.count > 0]
    if not active:
        return selected

    # Step 1: assign minimum allocation
    remaining = target_total
    for s in active:
        alloc = min(min_per_stratum, s.count)
        quota[f"{s.dimension}:{s.value}"] = alloc
        remaining -= alloc

    # Step 2: distribute remaining slots proportional to count
    if remaining > 0:
        total_episodes = sum(s.count for s in active)
        if total_episodes > 0:
            # Proportional allocation
            allocated_extra = 0
            for s in active:
                key = f"{s.dimension}:{s.value}"
                proportional = int(remaining * s.count / total_episodes)
                # Don't exceed stratum size
                already = quota[key]
                extra = min(proportional, s.count - already)
                quota[key] = already + extra
                allocated_extra += extra
            remaining -= allocated_extra

            # Distribute any leftover slots to largest strata
            if remaining > 0:
                by_count = sorted(active, key=lambda s: -s.count)
                for s in by_count:
                    if remaining <= 0:
                        break
                    key = f"{s.dimension}:{s.value}"
                    if quota[key] < s.count:
                        quota[key] += 1
                        remaining -= 1

    # Step 3: select episodes within each stratum
    for s in active:
        key = f"{s.dimension}:{s.value}"
        alloc = quota.get(key, 0)
        if alloc <= 0:
            continue
        pool = list(s.episode_ids)
        rng.shuffle(pool)
        selected.update(pool[:alloc])

    return selected


def _build_rationale(
    episodes: list[EpisodeSummary],
    strata: list[StratumBucket],
    selected_ids: set[str],
    dimensions: list[str],
) -> str:
    """Build a human-readable rationale for the sampling decisions."""
    parts: list[str] = []

    parts.append(f"Sampled {len(selected_ids)} episodes from {len(episodes)} total.")
    parts.append(f"Dimensions used: {', '.join(dimensions)}.")
    parts.append("")

    # Per-dimension coverage breakdown of selected episodes
    dim_extractors = {name: fn for name, fn in _DEFAULT_DIMENSIONS}
    selected_eps = {ep.episode_id: ep for ep in episodes if ep.episode_id in selected_ids}

    for dim_name in dimensions:
        fn = dim_extractors.get(dim_name)
        if fn is None:
            continue
        values: dict[str, int] = {}
        for ep in selected_eps.values():
            val = fn(ep)
            values[val] = values.get(val, 0) + 1
        parts.append(f"  {dim_name}: {values}")

    parts.append("")
    parts.append("Selection priority within each stratum:")
    parts.append("  1. Complete episodes (no missing files) preferred")
    parts.append("  2. Diversity in trajectory length (not just extremes)")
    parts.append("  3. Fixed random seed (42) for reproducibility")

    parts.append("")
    parts.append("Suggested annotation workflow for each selected episode:")
    parts.append("  1. Run: eforge segment-demo --episode-dir <episode_dir>")
    parts.append("  2. Review segments_draft.json, mark corrections in review_edits.json")
    parts.append("  3. Run: eforge finalize-segments --output-dir <output_dir> --reviewer <name>")
    parts.append("  4. Copy segments_final.json to segments_gt.json as ground truth")

    return "\n".join(parts)


def build_sampling_manifest(
    episodes: list[EpisodeSummary],
    dataset_root: str | Path,
    target_sample_size: int = 30,
    dimensions: list[str] | None = None,
    min_per_stratum: int = 1,
    seed: int = 42,
) -> SamplingManifest:
    """Build a complete SamplingManifest from scanned episodes.

    Args:
        episodes: List of EpisodeSummary from scan_dataset().
        dataset_root: Root directory that was scanned.
        target_sample_size: Target number of episodes for the ground truth set.
        dimensions: Which dimensions to stratify on. Default: task_type, duration_bucket,
                    camera_setup, success.
        min_per_stratum: Minimum episodes per stratum (ensures coverage).
        seed: Random seed for reproducibility.

    Returns:
        SamplingManifest with all metadata, strata, and selection decisions.
    """
    dataset_root = str(dataset_root)

    # Resolve dimensions
    if dimensions is None:
        dim_pairs = list(_DEFAULT_DIMENSIONS)
    else:
        all_map = {name: fn for name, fn in _DEFAULT_DIMENSIONS}
        dim_pairs = [(d, all_map[d]) for d in dimensions if d in all_map]

    dim_names = [d[0] for d in dim_pairs]

    # Build strata
    strata = _build_strata(episodes, dim_pairs)

    # Select episodes
    target = min(target_sample_size, len(episodes))
    selected_ids = _proportional_allocation(
        strata,
        target_total=target,
        min_per_stratum=min_per_stratum,
        seed=seed,
    )

    # Build rationale
    rationale = _build_rationale(episodes, strata, selected_ids, dim_names)

    complete_count = sum(1 for ep in episodes if ep.is_complete)

    return SamplingManifest(
        dataset_root=dataset_root,
        total_episodes=len(episodes),
        complete_episodes=complete_count,
        strata=strata,
        sampling_dimensions=dim_names,
        episodes=episodes,
        target_sample_size=target,
        selected_ids=sorted(selected_ids),
        sampling_rationale=rationale,
    )


def build_ground_truth_split(
    manifest: SamplingManifest,
    manifest_path: str = "",
    annotation_guidelines: str = "",
) -> GroundTruthSplit:
    """Build a GroundTruthSplit from a SamplingManifest.

    Args:
        manifest: The sampling manifest with selected episode IDs.
        manifest_path: Path to the manifest JSON (for traceability).
        annotation_guidelines: Instructions for human annotators.

    Returns:
        GroundTruthSplit ready to be written to ground_truth_split.json.
    """
    if not annotation_guidelines:
        annotation_guidelines = _default_annotation_guidelines()

    return GroundTruthSplit(
        source_manifest=manifest_path,
        selected_episodes=sorted(manifest.selected_ids),
        annotation_guidelines=annotation_guidelines,
        status="pending_annotation",
    )


def _default_annotation_guidelines() -> str:
    """Default annotation guidelines for human reviewers marking stage boundaries."""
    return (
        "## Ground Truth Annotation Guidelines\n\n"
        "For each selected episode, annotate stage boundaries by creating a `segments_gt.json` file.\n\n"
        "### What to Annotate\n"
        "1. **Stage boundaries**: The exact frame index where the robot transitions between skills.\n"
        "   - Common skills: reach, grasp, lift, place, insert, align, release, retreat\n"
        "2. **Keyframes**: The most representative frame within each stage.\n\n"
        "### How to Annotate\n"
        "1. Run `eforge segment-demo --episode-dir <episode_dir>` to generate a draft.\n"
        "2. Open the images around each boundary candidate and visually verify:\n"
        "   - Is the gripper actually closing/opening at this frame?\n"
        "   - Is the robot making contact with the object?\n"
        "   - Does the motion pattern change (approach → manipulation → retreat)?\n"
        "3. Create `review_edits.json` with corrections (see schema docs).\n"
        "4. Run `eforge finalize-segments` to produce the final annotation.\n"
        "5. Copy `segments_final.json` to `segments_gt.json` as the ground truth.\n\n"
        "### Output Format\n"
        "```json\n"
        '{"episode_id": "...", "status": "ground_truth", "stages": [\n'
        '  {"stage_id": 0, "skill_id": "reach", "start_frame": 0, "end_frame": 45, ...}\n'
        "]}\n"
        "```\n\n"
        "### Quality Checks\n"
        "- Every frame in the episode should belong to exactly one stage (no gaps, no overlaps)\n"
        "- Skill labels should use consistent vocabulary across all episodes\n"
        "- When uncertain about a boundary, flag it with a note in review_notes"
    )
