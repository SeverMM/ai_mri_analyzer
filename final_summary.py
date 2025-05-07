from __future__ import annotations

"""Generate a patient-friendly + professional summary from the CSV report.

The function defined here is called automatically from *main.py* once the CSV and
PDF reports are done.  It uploads the CSV content to the same OpenAI model used
elsewhere and saves the returned text into ``summary_<timestamp>.txt`` inside
*reports_dir*.
"""

import logging
import os
import time
from pathlib import Path
from typing import Tuple

import openai  # type: ignore

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Public helper
# --------------------------------------------------------------------------------------

def generate_final_summary(csv_path: Path, reports_dir: Path, model_name: str = "o3") -> Path:
    """Send *csv_path* to OpenAI ChatCompletion and save the narrative summary.

    Args:
        csv_path: Path to the CSV report that should be summarised.
        reports_dir: Directory where the summary text file will be written.
        model_name: OpenAI model to use (defaults to the same *mini* variant).

    Returns:
        The path of the generated summary text file.
    """

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set. Please create a .env file.")

    # Read CSV content
    csv_content = csv_path.read_text(encoding="utf-8")

    system_prompt = (
        "You are a highly experienced radiologist and medical writer. "
        "You will receive a CSV file that contains an AI-generated interpretation of an MRI study. "
        "You must produce two consecutive sections:\n\n"
        "1. A plain-language summary (≤ 250 words) that can be understood by a layperson.\n"
        "2. A detailed professional report providing nuanced findings, clinical significance, and actionable next steps "
        "for healthcare professionals. This section may use medical terminology, cite imaging sequences or slice numbers, "
        "and should end with clear recommendations for further imaging or management."
    )

    user_message = (
        "Here is the CSV report for the MRI study. Generate the two-part summary as instructed.\n\n" + csv_content
    )

    client = openai.OpenAI(api_key=openai_api_key)
    logger.info("Requesting final patient + professional summary from OpenAI (%s)…", model_name)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
    )

    summary_text = response.choices[0].message.content.strip()

    # Persist summary
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = reports_dir / f"summary_{timestamp}.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("Final textual summary saved to %s", summary_path)

    return summary_path 