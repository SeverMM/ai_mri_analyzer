"""Summarize raw model outputs stored in the *results* folder.

The summarization pipeline is intentionally simple for now:

1. Load every ``*.json`` file under *results_dir* – those originate from :pyfunc:`ai_mri_analyzer.batch.analyze_series`.
2. Group by *series_id* (the prefix before ``_batch`` in the filename).
3. For each series, aggregate:
   • findings – flattened list of strings (duplicates removed, order preserved)
   • impression – concatenated paragraphs separated by two new-lines
   • recommendations – same as impression
4. Produce a *study-level* summary by concatenating each series' impression & recommendations.

The output is two dictionaries:
    - ``series_summary`` – mapping *series_id* → dict with keys findings[], impression, recommendations
    - ``study_summary`` – overall findings/impression/recommendations across entire study

Later we can switch to LLM-based deduplication / rewriting if needed.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

__all__ = ["summarize_results"]


def _unique_preserve_order(items: List[str]) -> List[str]:
    """Return *items* deduplicated while preserving the first occurrence order."""

    seen = OrderedDict()
    for item in items:
        key = str(item)
        if key not in seen:
            seen[key] = item
    return list(seen.values())


def summarize_results(results_dir: Path) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Aggregate every JSON file in *results_dir*.

    Args:
        results_dir: Directory containing ``<series_id>_batchN.json`` files.

    Returns:
        Tuple ``(series_summary, study_summary)``.
    """

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    series_data: Dict[str, List[dict]] = {}

    for file in results_dir.glob("*.json"):
        if "_batch" not in file.stem:
            continue  # skip any future non-batch files

        series_id = file.stem.split("_batch")[0]

        try:
            obj = json.loads(file.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover – malformed JSON should be inspected manually
            continue

        series_data.setdefault(series_id, []).append(obj)

    # Aggregate per series ------------------------------------------------------------------------------------------
    series_summary: Dict[str, Dict[str, object]] = {}

    for sid, batches in series_data.items():
        all_findings: List[str] = []
        impressions: List[str] = []
        recs: List[str] = []

        for b in batches:
            # Findings might be list[str] or list[dict]. Convert dict to str.
            raw_findings = b.get("findings", [])
            if isinstance(raw_findings, list):
                norm = [str(f) if not isinstance(f, str) else f for f in raw_findings]
                all_findings.extend(norm)
            else:
                all_findings.append(str(raw_findings))

            if (imp := b.get("impression")):
                if isinstance(imp, list):
                    impressions.append(" ".join(str(x) for x in imp).strip())
                else:
                    impressions.append(str(imp).strip())

            if (r := b.get("recommendations")):
                if isinstance(r, list):
                    recs.append(" ".join(str(x) for x in r).strip())
                else:
                    recs.append(str(r).strip())

        series_summary[sid] = {
            "findings": _unique_preserve_order(all_findings),
            "impression": "\n\n".join(impressions),
            "recommendations": "\n\n".join(recs),
        }

    # Study-level aggregation --------------------------------------------------------------------------------------
    study_findings: List[str] = []
    study_impressions: List[str] = []
    study_recs: List[str] = []

    for data in series_summary.values():
        study_findings.extend([str(f) for f in data["findings"]])  # type: ignore[arg-type]
        study_impressions.append(data["impression"])  # type: ignore[arg-type]
        study_recs.append(data["recommendations"])  # type: ignore[arg-type]

    study_summary = {
        "findings": _unique_preserve_order(study_findings),
        "impression": "\n\n".join(study_impressions),
        "recommendations": "\n\n".join(study_recs),
    }

    return series_summary, study_summary 