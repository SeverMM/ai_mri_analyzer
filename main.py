"""CLI entrypoint for ai_mri_analyzer.

Usage examples:
    # Analyse only first 3 images per series for a dry-run
    python -m ai_mri_analyzer /path/to/images --sample 3

    # Full run (default batch size = config.toml)
    python -m ai_mri_analyzer /path/to/images
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List

from .batch import analyze_series
from .series import infer_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("ai_mri_analyzer")


def _collect_files(root: Path) -> List[Path]:
    """Recursively collect all DICOM/JPEG/PNG files inside *root*."""
    patterns = ("**/*.dcm", "**/*.jpg", "**/*.jpeg", "**/*.png")
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(files)


def parse_args() -> argparse.Namespace:  # noqa: D401 (docstring style)
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AI-powered MRI analyzer")
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing DICOM or JPEG exports",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If given, only the first N images of each series are processed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (images per OpenAI request).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent OpenAI requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per batch upon rate-limit or network errors.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="Global requests-per-minute limit (0 = no limit).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (defaults to package path).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip automatic summarisation and report generation (CSV/PDF).",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        metavar="SERIES_ID",
        help="If provided, only the given series identifiers are analysed (e.g. IMG-0003 IMG-0005).",
    )
    parser.add_argument(
        "--prev-flag",
        type=str,
        default="abnormality",
        help="Preliminary AI finding to confirm or refute (injected into the prompt).",
    )
    parser.add_argument(
        "--patient-context",
        type=str,
        default="",
        metavar="TEXT",
        help="Short demographics / relevant history (e.g. 'Male, 74 y, treated prostate cancer').",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:  # noqa: D401
    """Main async driver — groups series and dispatches analysis."""
    all_files = _collect_files(args.image_dir)
    if not all_files:
        logger.error("No image files found in %s", args.image_dir)
        return

    logger.info("Found %d image files. Grouping into series…", len(all_files))
    series_map = infer_series(all_files)

    if args.series:
        include_set = {s.strip() for s in args.series}
        series_map = {sid: paths for sid, paths in series_map.items() if sid in include_set}

    logger.info("Identified %d series after filtering.", len(series_map))

    tasks = []
    for series_id, paths in series_map.items():
        tasks.append(
            analyze_series(
                series_id,
                paths,
                batch_size=args.batch_size or 20,
                sample_limit=args.sample,
                config_path=args.config,
                patient_context=args.patient_context,
                previous_ai_flag=args.prev_flag,
                max_concurrent=args.max_concurrent,
                max_retries=args.max_retries,
                requests_per_minute=args.rpm,
            )
        )

    await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Automatic summarise + report generation
    # ------------------------------------------------------------------
    if args.skip_report:
        logger.info("--skip-report flag set – not generating summary/report.")
        return

    # Determine results_dir / reports_dir from config (or defaults)
    import tomllib
    if args.config is None:
        config_path = Path(__file__).parent / "config.toml"
    else:
        config_path = args.config

    config = {}
    if config_path.exists():
        try:
            config = tomllib.loads(config_path.read_text())
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to parse config (%s) – %s", config_path, exc)

    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))

    logger.info("Generating aggregate summary and report…")
    from .summarize import summarize_results
    from .report import export_reports

    series_summary, study_summary = summarize_results(results_dir)
    pdf_path, csv_path = export_reports(series_summary, study_summary, out_dir=reports_dir)

    logger.info("Summary and report written to %s", reports_dir)

    # ------------------------------------------------------------------
    # Patient-friendly + professional summary via OpenAI
    # ------------------------------------------------------------------
    try:
        from .final_summary import generate_final_summary

        summary_model_cfg = config.get("openai", {}).get("summary_model", "o3")
        summary_txt = generate_final_summary(csv_path, reports_dir, model_name=summary_model_cfg)
        logger.info("Generated plain + professional summary at %s", summary_txt.name)
    except Exception as exc:  # pragma: no cover – we don't want the whole run to crash
        logger.error("Failed to generate final summary – %s", exc)


def main() -> None:  # noqa: D401
    """Sync wrapper around :pyfunc:`_run`."""
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main() 