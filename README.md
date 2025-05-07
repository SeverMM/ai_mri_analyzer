# AI MRI Analyzer

`ai_mri_analyzer` is a minimal, CLI-driven tool that batches MRI slices (DICOM/JPEG/PNG) into small groups and asks GPT-4o Vision for radiological findings. It then aggregates the model answers and produces both a CSV and a multi-page PDF report.

> ⚠️ **DISCLAIMER**  This project is for research / educational purposes only.  It is **not** a medical device, does **not** provide diagnoses and must **never** be used as a substitute for professional radiological evaluation.

---

## Features

| Stage                     | Functionality |
|---------------------------|---------------|
| Ingest                    | Loads DICOM (pydicom) or JPEG/PNG (Pillow) to NumPy arrays, extracts basic metadata |
| Series grouping           | Groups files by `SeriesInstanceUID` or by filename pattern `IMG-0003-xxxxx.jpg` |
| Batching                  | ≤ 20 images per OpenAI request; async with concurrency control & exponential back-off |
| Prompting                 | Enhanced per-series prompt with demographics, sequence type and clinical question.  Returns strict JSON schema with `findings[]/impression/recommendations` plus confidence & suspicion level |
| Resume                    | Already-processed batches are skipped on re-run |
| Summarise                 | Deduplicates findings; concatenates impressions/recommendations per series & study |
| Report                    | Generates CSV + PDF (`report_<timestamp>.pdf`) via ReportLab |
| Final summary             | Sends CSV to OpenAI and writes a layperson + professional text summary |
|                           | (uses advanced reasoning model **o3** by default) |

---

## Installation (Windows 11 / macOS / Linux)

```powershell
# 1. Clone / download the repo
cd path\to\folder

# 2. Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell
# source .venv/bin/activate       # macOS/Linux

# 3. Install dependencies
pip install -r ai_mri_analyzer\requirements.txt
```

If VS Code shows *missing import* warnings, ensure it's using the same interpreter (the virtual-env above).

### OpenAI API key

Option A – **.env file** (recommended):
```dotenv
OPENAI_API_KEY=sk-…yourkey…
```
Place this file in the project root. It is automatically loaded on startup via *python-dotenv*.

Option B – environment variable:
```powershell
setx OPENAI_API_KEY "sk-…yourkey…"   # permanent for user
$Env:OPENAI_API_KEY = "sk-…yourkey…"  # current session only
```

Option C – edit `ai_mri_analyzer\config.toml` if you wish to override the model only:
```toml
[openai]
# key is picked up from environment or .env
model = "gpt-4o"
summary_model = "o3"   # model used for the final text summary
```

---

## Preparing images
Place all DICOM or exported JPEG/PNG slices in a folder, e.g.:
```
C:\MRI\export\
    IMG-0001-0001.jpg
    IMG-0001-0002.jpg
    …
```
Sub-folders are fine; search is recursive.

---

## Command-line usage

```powershell
python -m ai_mri_analyzer IMAGE_DIR [options]
```

| Option                 | Default | Description |
|------------------------|---------|-------------|
| `--sample N`           | *none*  | Only the first **N** images of each series (cheap dry-run) |
| `--batch-size N`       | 20      | Images per OpenAI request (1-20) |
| `--max-concurrent N`   | 5       | Parallel OpenAI requests |
| `--max-retries N`      | 3       | Retries per batch on rate-limit/network errors |
| `--config PATH`        | internal `config.toml` | Custom config location |
| `--rpm N`              | 60      | Global **requests-per-minute** limit (0 = disable limiter) |
| `--series ID …`        | *none*  | Only analyse the specified series identifiers (e.g. `IMG-0003 IMG-0022`) |
| `--prev-flag TEXT`     | "abnormality" | Clinical question injected into the prompt (confirm / refute) |
| `--patient-context TEXT` | *empty* | Demographics / relevant history to include in the prompt |
| `--skip-report`        | false   | Skip automatic CSV/PDF generation |

### Examples

• Quick test on first 3 images of each series:
```powershell
python -m ai_mri_analyzer C:\MRI\export --sample 3 --batch-size 5
```

• Full analysis (Tier-2 key) with 4 concurrent requests:
```powershell
python -m ai_mri_analyzer C:\MRI\export --batch-size 20 --max-concurrent 4 --rpm 200
```

• Tier-1 key safe settings (3 requests/min, 2 000 tokens/min):
```powershell
python -m ai_mri_analyzer C:\MRI\export --batch-size 5 --rpm 3
```

• Focus on three series flagged during a preliminary sweep:
```powershell
python -m ai_mri_analyzer C:\MRI\export \
       --series IMG-0003 IMG-0005 IMG-0022 \
       --prev-flag "possible extraprostatic extension" \
       --patient-context "Male, 47 y, treated prostate cancer; rising PSA" \
       --batch-size 20
```

• Resume after interruption (same command – processed batches are skipped).

---

## Outputs
```
results/               # one JSON per batch (<series>_batchN.json)
reports/
    report_YYYYMMDD_HHMMSS.csv
    report_YYYYMMDD_HHMMSS.pdf
    summary_YYYYMMDD_HHMMSS.txt
```

### PDF layout
1. Study-level impression & recommendations (page 1)  
2. One page per series: findings table, impression, recommendations (plus optional thumbnail)

---

## Run the tests
```powershell
pytest -q
```

---

## Tips for getting higher-quality model answers

1. **Include concise clinical context** – age, sex, known diagnosis, symptoms, prior treatments.  Example: `patient_context="Male, 47 y, treated prostate cancer; rising PSA"`.
2. **Specify series modality & plane** – our prompt already passes the SeriesDescription, but ensure filenames keep clues (e.g. T2_AX).  The model will reason better.
3. **Focus the task** – replace the default `task_instruction` with a narrower question (e.g. "Assess lymph-node burden and seminal-vesicle invasion").
4. **Ask for differential** – add "Also list differential diagnoses if findings are ambiguous."
5. **Use follow-up batches** – after initial run, feed the study-level summary back to the model with further questions.
6. **Guard against anchoring bias** – if you supply a suspected diagnosis, ask the model to *also* consider alternatives to avoid over-confidence.
7. **Token budget** – smaller `batch_size` (5-10) often yields more detailed per-slice commentary because generation isn't heavily truncated.

---

## Roadmap
- Better DICOM windowing & orientation handling
- LLM post-processing of impressions
- Streamlit web UI
- Support for Claude & Gemini APIs 