"""Sampling module — dataset scanning, stratified sampling, ground truth split.

Phase 1 of the data-centric segmentation workflow:
1. Scan dataset → episode metadata
2. Stratify → representative subset
3. Output sampling_manifest.json and ground_truth_split.json
"""
