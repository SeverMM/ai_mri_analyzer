"""Batch processing module for ai_mri_analyzer.

This module is responsible for:
1. Converting image paths to base-64 data URIs (required by OpenAI vision models).
2. Chunking images into batches (default ≤ 20) to stay well within model limits.
3. Building the prompt using :pydata:`ai_mri_analyzer.prompts` templates.
4. Sending *stream=True* calls to the OpenAI Chat Completions endpoint.
5. Persisting raw JSON responses under *results_dir*.

The functions are designed to be used from :pymod:`ai_mri_analyzer.main` but remain composable
in case we later add a Streamlit UI or other front-ends.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import tomllib

import openai  # type: ignore

from .prompts import SYSTEM_PROMPT, USER_TEMPLATE

# --------------------------------------------------------------------------------------
# Constants & utilities
# --------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _chunked(seq: Sequence[Path], size: int) -> Iterable[List[Path]]:
    """Yield *size*-sized chunks from *seq*."""

    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def _img_path_to_data_uri(path: Path) -> str:
    """Encode *path* to a data URI suitable for OpenAI vision requests."""

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    with path.open("rb") as fp:
        b64 = base64.b64encode(fp.read()).decode()

    return f"data:{mime};base64,{b64}"


def _prepare_user_content(
    *,
    series_id: str,
    sequence_type: str,
    slice_count: int,
    patient_context: str,
    previous_ai_flag: str,
    images: List[Path],
) -> List[Any]:
    """Build the *content* array for the user message.

    The function fills the :data:`ai_mri_analyzer.prompts.USER_TEMPLATE` with the
    provided parameters and constructs the list structure expected by the
    OpenAI vision chat endpoint (text followed by image URLs).

    Args:
        series_id: Identifier such as ``IMG-0003``.
        sequence_type: Human-readable sequence description (e.g. *T2-weighted axial pelvis*).
        slice_count: Number of slices/images attached to this request.
        patient_context: Short demographics / relevant history.
        previous_ai_flag: Condition or abnormality detected during the preliminary
            AI sweep that we now want to confirm or refute.
        images: Image paths that will be converted to data URIs and appended to
            the user message.

    Returns:
        List compliant with the *content* schema for Chat Completions (first
        element must be a ``{"type": "text", "text": ...}`` item followed by
        ``{"type": "image_url", ...}`` items).
    """

    # JSON schema block that the model should replicate. Keeping this in a
    # dedicated variable makes the surrounding f-string easier to read.
    json_schema = (
        "{\n"
        "  \"findings\": [\n"
        "    {\n"
        "      \"slice_index\": int,\n"
        "      \"anatomical_location\": str,\n"
        "      \"description\": str,\n"
        "      \"suspicion_level\": \"benign|indeterminate|suspicious|highly_suspicious\",\n"
        "      \"confidence\": int\n"
        "    }\n"
        "  ],\n"
        "  \"impression\": str,\n"
        "  \"recommendations\": str\n"
        "}"
    )

    text_block = USER_TEMPLATE.format(
        series_id=series_id,
        sequence_type=sequence_type,
        slice_count=slice_count,
        patient_context=patient_context or "N/A",
        previous_ai_flag=previous_ai_flag,
        json_schema=json_schema,
    )

    content: List[Any] = [{"type": "text", "text": text_block}]

    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _img_path_to_data_uri(img)}})

    return content


def _infer_sequence_type(paths: List[Path]) -> str:
    """Best-effort attempt to derive a human-readable sequence description.

    The function loads the first DICOM file it encounters and checks
    ``SeriesDescription`` or ``BodyPartExamined`` tags. If nothing can be
    inferred, the generic placeholder *Unknown sequence type* is returned.
    """

    from .ingest import load_image  # Local import to avoid a heavy dependency when not needed

    for p in paths:
        if p.suffix.lower() == ".dcm":
            try:
                meta = load_image(p)["metadata"]
            except Exception:  # pragma: no cover – we fall back to default later
                continue

            for key in ("SeriesDescription", "BodyPartExamined", "Modality"):
                value = meta.get(key)
                if value:
                    return str(value)

    return "Unknown sequence type"


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


async def analyze_series(
    series_id: str,
    image_paths: List[Path],
    *,
    batch_size: int = 20,
    sample_limit: int | None = None,
    config_path: Path | None = None,
    patient_context: str = "",
    previous_ai_flag: str = "abnormality",
    sequence_type: str | None = None,
    max_concurrent: int = 5,
    max_retries: int = 3,
    requests_per_minute: int = 60,
) -> None:
    """Analyse *image_paths* in batches and persist JSON responses.

    Args:
        series_id: Unique identifier for the MRI series.
        image_paths: All images that belong to the *series_id*.
        batch_size: Maximum images per OpenAI request (≤ 20 recommended).
        sample_limit: If provided, only the *first* *sample_limit* images of this list are used.
        config_path: Path to ``config.toml`` (default: locate next to *main.py*).
        patient_context: String injected into USER prompt with demographic info etc.
        previous_ai_flag: Preliminary AI finding we want to confirm/refute.
        sequence_type: MRI sequence description. If *None*, the function will
            attempt to derive it from DICOM metadata.
        max_concurrent: Maximum concurrent requests to OpenAI.
        max_retries: Maximum number of retries for each batch.
        requests_per_minute: Maximum number of requests per minute.
    """

    # Load config ----------------------------------------------------------------------------------------------------
    if config_path is None:
        # Default: use <package_root>/config.toml
        config_path = Path(__file__).parent / "config.toml"

    config: Dict[str, Any]
    try:
        config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
    except Exception as exc:  # pragma: no cover – safe-guard any format error
        logger.warning("Failed to parse config file %s – %s", config_path, exc)
        config = {}

    openai_api_key = os.getenv("OPENAI_API_KEY", config.get("openai", {}).get("api_key"))
    if not openai_api_key:
        raise ValueError("OpenAI API key not provided via env var or config.toml")

    model_name = config.get("openai", {}).get("model", "gpt-4o-mini")

    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Apply *sample_limit* -------------------------------------------------------------------------------------------
    if sample_limit is not None:
        image_paths = image_paths[:sample_limit]

    # Derive *sequence_type* if not provided ------------------------------------------------------------------------
    if sequence_type is None:
        sequence_type = _infer_sequence_type(image_paths)

    # Guard rails ----------------------------------------------------------------------------------------------------
    if batch_size < 1 or batch_size > 20:
        raise ValueError("batch_size must be between 1 and 20 for GPT-4o vision usage")

    # Prepare client -------------------------------------------------------------------------------------------------
    client = openai.AsyncOpenAI(api_key=openai_api_key)

    # Concurrency semaphore
    sem = asyncio.Semaphore(max_concurrent)

    # Iterate batches ------------------------------------------------------------------------------------------------
    tasks = []
    for batch_idx, batch_paths in enumerate(_chunked(image_paths, batch_size), start=1):
        tasks.append(
            _run_single_batch(
                client,
                model_name,
                series_id,
                batch_idx,
                batch_paths,
                sequence_type,
                patient_context,
                previous_ai_flag,
                results_dir,
                sem,
                max_retries,
                requests_per_minute,
            )
        )

    await asyncio.gather(*tasks)


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


async def _run_single_batch(
    client: openai.AsyncOpenAI,
    model_name: str,
    series_id: str,
    batch_idx: int,
    batch_paths: List[Path],
    sequence_type: str,
    patient_context: str,
    previous_ai_flag: str,
    results_dir: Path,
    sem: asyncio.Semaphore,
    max_retries: int,
    requests_per_minute: int,
) -> None:
    """Send *one* batch of images to the model.

    The raw streamed response is reassembled into text and saved as JSON under
    ``<results_dir>/<series_id>_batch<batch_idx>.json``.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _prepare_user_content(
                series_id=series_id,
                sequence_type=sequence_type,
                slice_count=len(batch_paths),
                patient_context=patient_context,
                previous_ai_flag=previous_ai_flag,
                images=batch_paths,
            ),
        },
    ]

    outfile = results_dir / f"{series_id}_batch{batch_idx}.json"

    # Resume support -- skip if file already exists -----------------------------------------------------------------
    if outfile.exists():
        logger.info("Skipping batch %s for series %s – result already exists at %s", batch_idx, series_id, outfile)
        return

    async with sem:
        # Simple global rate limiter -----------------------------------------------------------------
        await _respect_rpm(requests_per_minute)
        attempt = 0
        backoff = 2  # seconds
        while True:
            attempt += 1
            try:
                logger.info(
                    "Sending batch %s (%d images) for series %s to OpenAI model %s (attempt %d)",
                    batch_idx,
                    len(batch_paths),
                    series_id,
                    model_name,
                    attempt,
                )

                stream = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    max_completion_tokens=1024,  # generous default – adjust as needed
                    response_format={"type": "json_object"},
                )

                # Re-assemble streamed chunks into one JSON string --------------------------------------------------
                full_content = ""
                async for chunk in stream:  # pragma: no cover – streamed chunks
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_content += delta.content

                # Persist raw response -----------------------------------------------------------------------------
                outfile.write_text(full_content, encoding="utf-8")
                logger.info("Saved response for series %s batch %s -> %s", series_id, batch_idx, outfile)

                # Basic validation ---------------------------------------------------------------------------------
                try:
                    data = json.loads(full_content)
                    missing = [k for k in ("findings", "impression", "recommendations") if k not in data]
                    if missing:
                        raise KeyError(", ".join(missing))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Model response validation issue (%s). File saved for manual inspection.", exc
                    )
                return  # success

            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
                if attempt > max_retries:
                    logger.error("Batch %s failed after %d retries – %s", batch_idx, max_retries, exc)
                    return

                sleep_for = backoff * attempt
                logger.warning(
                    "OpenAI API error (%s). Retrying batch %s in %.1f s (attempt %d/%d)…",
                    exc.__class__.__name__,
                    batch_idx,
                    sleep_for,
                    attempt,
                    max_retries,
                )
                await asyncio.sleep(sleep_for)


# --------------------------------------------------------------------------------------
# Rate limiting helper (simple fixed-window)
# --------------------------------------------------------------------------------------


_LAST_CALL_TS: float | None = None
_RPM_LOCK = asyncio.Lock()


async def _respect_rpm(rpm: int) -> None:
    """Await until making the next call keeps us under *rpm* limit."""

    global _LAST_CALL_TS  # noqa: PLW0603

    if rpm <= 0:
        return  # disabled

    min_interval = 60.0 / rpm

    async with _RPM_LOCK:
        now = time.monotonic()
        if _LAST_CALL_TS is None:
            _LAST_CALL_TS = now
            return

        wait_for = _LAST_CALL_TS + min_interval - now
        if wait_for > 0:
            await asyncio.sleep(wait_for)
            _LAST_CALL_TS = time.monotonic()
        else:
            _LAST_CALL_TS = now 