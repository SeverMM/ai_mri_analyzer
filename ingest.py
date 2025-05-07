"""Ingest module for ai_mri_analyzer.

Loads DICOM or JPEG files and converts them into numpy arrays along with essential metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

try:
    import pydicom  # type: ignore
    from pydicom.errors import InvalidDicomError  # type: ignore
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore
    InvalidDicomError = Exception  # type: ignore


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------------------

def load_image(file_path: Path) -> Dict[str, Any]:
    """Load a single DICOM or JPEG file.

    Args:
        file_path: Path to the image file.

    Returns:
        Dictionary containing:
            - "pixels": numpy.ndarray of the image pixel data (uint8)
            - "metadata": dict with selected metadata fields

    Raises:
        FileNotFoundError: If the provided file does not exist.
        ValueError: If the file extension is unsupported or reading fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".dcm":
        return _load_dicom(file_path)
    elif suffix in {".jpg", ".jpeg", ".png"}:
        return _load_jpeg(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _load_dicom(file_path: Path) -> Dict[str, Any]:
    """Load DICOM file using pydicom.

    Converts pixel data to 8-bit grayscale (if needed) and extracts common tags.
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files")

    try:
        ds = pydicom.dcmread(str(file_path))
    except InvalidDicomError as exc:
        raise ValueError(f"Invalid DICOM file: {file_path}") from exc

    # Extract pixel data
    pixel_array = ds.pixel_array.astype(np.float32)

    # Windowing or scaling if PhotometricInterpretation not MONOCHROME? Keep simple for now.
    pixel_array = _scale_to_uint8(pixel_array)

    metadata: Dict[str, Any] = {
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
        "SeriesDescription": getattr(ds, "SeriesDescription", None),
        "BodyPartExamined": getattr(ds, "BodyPartExamined", None),
        "Modality": getattr(ds, "Modality", None),
        "EchoTime": getattr(ds, "EchoTime", None),
        "PatientSex": getattr(ds, "PatientSex", None),
        "PatientAge": getattr(ds, "PatientAge", None),
        "StudyDescription": getattr(ds, "StudyDescription", None),
        "filepath": str(file_path),
    }

    return {"pixels": pixel_array, "metadata": metadata}


def _load_jpeg(file_path: Path) -> Dict[str, Any]:
    """Load JPEG/PNG file using PIL.

    Converts to numpy uint8 array and constructs minimal metadata.
    """
    with Image.open(file_path) as img:
        # Ensure RGB or L; convert to L (grayscale) to reduce size
        if img.mode != "L":
            img = img.convert("L")
        pixel_array = np.array(img, dtype=np.uint8)

    metadata = {
        "SeriesInstanceUID": None,
        "SeriesDescription": None,
        "BodyPartExamined": None,
        "Modality": "JPEG",
        "EchoTime": None,
        "filepath": str(file_path),
    }

    return {"pixels": pixel_array, "metadata": metadata}


def _scale_to_uint8(array: np.ndarray) -> np.ndarray:
    """Scale any numeric array to uint8 range [0, 255]."""
    if array.dtype == np.uint8:
        return array

    array_min = float(array.min())
    array_max = float(array.max())
    if array_max == array_min:
        return np.zeros_like(array, dtype=np.uint8)

    scaled = (array - array_min) / (array_max - array_min) * 255.0
    return scaled.astype(np.uint8) 