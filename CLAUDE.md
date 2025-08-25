# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI MRI Analyzer is a CLI tool that analyzes MRI images (DICOM/JPEG/PNG) using OpenAI's GPT-4o Vision API to generate radiological findings reports. It batches images, processes them through the AI model, and produces CSV and PDF reports.

**Important**: This is for research/educational purposes only, not for medical diagnosis.

## Key Commands

### Installation & Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (use .env file)
echo "OPENAI_API_KEY=sk-..." > .env
```

### Running the Application
```bash
# Run as module
python -m ai_mri_analyzer IMAGE_DIR [options]

# Quick test (first 3 images per series)
python -m ai_mri_analyzer /path/to/images --sample 3

# Full analysis with custom settings
python -m ai_mri_analyzer /path/to/images --batch-size 20 --max-concurrent 4
```

### Testing
```bash
# Run tests
pytest -q

# Run specific test
pytest path/to/test.py::test_function
```

## Architecture & Processing Pipeline

The application follows a linear processing pipeline:

1. **Ingest** (`ingest.py`): Loads DICOM/JPEG/PNG files into NumPy arrays with metadata extraction
2. **Series Grouping** (`series.py`): Groups images by SeriesInstanceUID or filename patterns
3. **Batch Processing** (`batch.py`): 
   - Chunks images into batches (≤20 per request)
   - Converts to base64 data URIs
   - Sends async requests to OpenAI API with exponential backoff
   - Stores results as JSON in `results/`
4. **Summarization** (`summarize.py`): Aggregates findings per series and study level
5. **Report Generation** (`report.py`): Creates CSV and PDF reports using ReportLab
6. **Final Summary** (`final_summary.py`): Generates layperson/professional text summaries using o3 model

### Key Design Patterns

- **Async Processing**: Uses `asyncio` for concurrent API requests with rate limiting
- **Resume Capability**: Skips already-processed batches on re-run (checks `results/*.json`)
- **Structured Output**: Enforces JSON schema for model responses (findings/impression/recommendations)
- **Configuration**: Uses TOML config (`config.toml`) for model settings
- **CLI Arguments**: Extensive argparse configuration in `main.py` for runtime options

### Module Responsibilities

- `main.py`: CLI entrypoint, argument parsing, orchestration
- `batch.py`: OpenAI API interaction, batching logic, retry mechanisms
- `ingest.py`: File loading (DICOM via pydicom, images via Pillow)
- `series.py`: Image grouping logic by series
- `prompts.py`: System and user prompt templates for the AI model
- `summarize.py`: Deduplication and aggregation of findings
- `report.py`: PDF/CSV generation with ReportLab
- `final_summary.py`: Text summary generation

### Output Structure
```
results/               # JSON files per batch (<series>_batchN.json)
reports/
    report_YYYYMMDD_HHMMSS.csv
    report_YYYYMMDD_HHMMSS.pdf
    summary_YYYYMMDD_HHMMSS.txt
```

## Important Configuration

- OpenAI models configured in `config.toml`:
  - `model`: Main analysis model (default: gpt-4o)
  - `summary_model`: Final summary model (default: o3)
- Rate limiting via `--rpm` flag (requests per minute)
- Batch size adjustable via `--batch-size` (1-20 images)
- Concurrent requests via `--max-concurrent`

## Development Notes

- The codebase uses type hints throughout
- Logging configured at INFO level by default
- Error handling includes exponential backoff for API rate limits
- Images are processed as base64 data URIs for OpenAI Vision API
- PDF generation uses ReportLab with dynamic page layout per series