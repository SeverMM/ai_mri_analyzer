"""Series grouping logic for ai_mri_analyzer."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

__all__ = ["infer_series"]


SERIES_PATTERN = re.compile(r"^([^-]+-[^-]+)")


def _series_key_from_filename(file_path: Path) -> str | None:
    """Attempt to extract series key from filename pattern.

    We expect patterns like `IMG-0003-02543.jpg` -> `IMG-0003`.
    """
    match = SERIES_PATTERN.match(file_path.stem)
    if match:
        return match.group(1)
    return None


def infer_series(files: List[Path]) -> Dict[str, List[Path]]:
    """Group files into series based on DICOM metadata or filename pattern.

    Args:
        files: List of file paths to images (DICOM or JPEG).

    Returns:
        Mapping from series identifier to list of file paths.

    Notes:
        The priority for identifying a series is:
        1. DICOM `SeriesInstanceUID` (if available via sidecar .dcm file in list)
        2. Substring between first two dashes in filename (e.g., IMG-0003)
        3. Fallback to a `default` key collecting ungroupable files.
    """
    from .ingest import load_image  # Local import to avoid circular deps

    series_map: Dict[str, List[Path]] = {}

    for file_path in files:
        series_id: str | None = None

        if file_path.suffix.lower() == ".dcm":
            try:
                data = load_image(file_path)
                uid = data["metadata"].get("SeriesInstanceUID")
                if uid:
                    series_id = str(uid)
            except Exception:  # pragma: no cover
                # If DICOM read fails, fallback to filename pattern
                series_id = None

        # If not found yet, try filename pattern
        if series_id is None:
            maybe_key = _series_key_from_filename(file_path)
            if maybe_key is not None:
                series_id = maybe_key

        # Final fallback
        if series_id is None:
            series_id = "default"

        series_map.setdefault(series_id, []).append(file_path)

    # Sort each list of paths for deterministic ordering
    for path_list in series_map.values():
        path_list.sort()

    return series_map 