from __future__ import annotations

"""ai_mri_analyzer package init.

Loads environment variables from a local .env file (if present).  This makes the
OpenAI API key available transparently across all sub-modules.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if it exists
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False) 