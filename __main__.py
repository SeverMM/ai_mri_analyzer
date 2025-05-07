"""Package entrypoint.

Allows running the tool with:
    python -m ai_mri_analyzer <image_dir> [options]
"""
from __future__ import annotations

from .main import main

if __name__ == "__main__":
    main() 