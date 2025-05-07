"""Generate PDF and CSV reports from summarised study data.

Only *ReportLab* is required for the PDF.  The PDF layout is minimal: study-level fields
first, followed by one page per series with a table of findings.  Thumbnails are optional
and only included if ``thumbnail_map`` is provided.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors

logger = logging.getLogger(__name__)

__all__ = ["export_reports"]


def export_reports(
    series_summary: Mapping[str, Mapping[str, object]],
    study_summary: Mapping[str, object],
    *,
    out_dir: Path,
    thumbnail_map: Dict[str, Path] | None = None,
) -> Tuple[Path, Path]:
    """Write ``report_<timestamp>.pdf`` and ``report_<timestamp>.csv`` to *out_dir*.

    Args:
        series_summary: Output from :pyfunc:`ai_mri_analyzer.summarize.summarize_results`.
        study_summary: Same as above (study-level aggregate).
        out_dir: Directory to write the files.
        thumbnail_map: Optional mapping *series_id* → path to thumbnail image (PNG/JPG).

    Returns:
        Tuple ``(pdf_path, csv_path)``.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"report_{timestamp}.csv"
    pdf_path = out_dir / f"report_{timestamp}.pdf"

    _write_csv(csv_path, series_summary, study_summary)
    _write_pdf(pdf_path, series_summary, study_summary, thumbnail_map)

    logger.info("Reports saved: %s, %s", pdf_path.name, csv_path.name)

    return pdf_path, csv_path


# --------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------


def _write_csv(
    path: Path,
    series_summary: Mapping[str, Mapping[str, object]],
    study_summary: Mapping[str, object],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)

        # Study-level first
        writer.writerow(["Study Impression"])
        writer.writerow([study_summary["impression"]])  # type: ignore[index]
        writer.writerow([])

        writer.writerow(["Study Recommendations"])
        writer.writerow([study_summary["recommendations"]])  # type: ignore[index]
        writer.writerow([])

        # Per-series rows
        writer.writerow(["Series ID", "Findings", "Impression", "Recommendations"])
        for sid, data in series_summary.items():
            writer.writerow(
                [
                    sid,
                    " | ".join(data["findings"]),  # type: ignore[index]
                    data["impression"],  # type: ignore[index]
                    data["recommendations"],  # type: ignore[index]
                ]
            )


# --------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------


def _write_pdf(
    path: Path,
    series_summary: Mapping[str, Mapping[str, object]],
    study_summary: Mapping[str, object],
    thumbnail_map: Dict[str, Path] | None,
) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    normal = ParagraphStyle("normal", fontSize=10, leading=12)
    header = ParagraphStyle("header", fontSize=14, leading=16, spaceAfter=12)

    def add_paragraph(text: str, style: ParagraphStyle, x: float, y: float) -> float:
        para = Paragraph(text.replace("\n", "<br/>"), style)
        w, h = para.wrap(width - 2 * x, height)
        para.drawOn(c, x, y - h)
        return y - h - 0.4 * cm  # return new y

    # First page – study-level summary -----------------------------------------------------------------------------
    y_pos = height - 2 * cm
    y_pos = add_paragraph("AI MRI Study Report", header, 2 * cm, y_pos)
    y_pos = add_paragraph("Impression", header, 2 * cm, y_pos)
    y_pos = add_paragraph(study_summary["impression"], normal, 2 * cm, y_pos)  # type: ignore[index]
    y_pos = add_paragraph("Recommendations", header, 2 * cm, y_pos)
    y_pos = add_paragraph(study_summary["recommendations"], normal, 2 * cm, y_pos)  # type: ignore[index]

    c.showPage()

    # Subsequent pages – per series ---------------------------------------------------------------------------------
    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])

    for sid, data in series_summary.items():
        y_pos = height - 2 * cm
        y_pos = add_paragraph(f"Series {sid}", header, 2 * cm, y_pos)

        # Optional thumbnail on right side
        thumb_path = thumbnail_map.get(sid) if thumbnail_map else None
        if thumb_path and thumb_path.exists():
            img_w = 6 * cm
            img_h = 6 * cm
            c.drawImage(str(thumb_path), width - img_w - 2 * cm, y_pos - img_h, img_w, img_h)

        # Findings table
        findings = data["findings"]  # type: ignore[index]
        table_data: List[List[str]] = [["#", "Finding"]]
        table_data.extend([[str(i + 1), f] for i, f in enumerate(findings)])

        tbl = Table(table_data, colWidths=[1.5 * cm, width - 5 * cm])
        tbl.setStyle(table_style)

        tbl_w, tbl_h = tbl.wrap(width - 4 * cm, y_pos)
        tbl.drawOn(c, 2 * cm, y_pos - tbl_h)

        y_pos = y_pos - tbl_h - 1 * cm

        # Impression + recommendations
        y_pos = add_paragraph("Impression", header, 2 * cm, y_pos)
        y_pos = add_paragraph(data["impression"], normal, 2 * cm, y_pos)  # type: ignore[index]
        y_pos = add_paragraph("Recommendations", header, 2 * cm, y_pos)
        y_pos = add_paragraph(data["recommendations"], normal, 2 * cm, y_pos)  # type: ignore[index]

        c.showPage()

    c.save() 