# LLM PDF Quiz Generator

A command line tool that generates educational quizzes from PDF documents using a local instruction-tuned language model. It extracts text from a PDF, falls back to OCR for scanned documents, asks the model to create quiz questions, and saves the result as JSON and PDF.


## Features

- PDF text extraction with `pypdf`
- OCR fallback with `pdf2image` and Tesseract
- Local Hugging Face model generation, defaulting to `Qwen/Qwen2.5-7B-Instruct`
- Question types: single choice, multiple choice, and open questions
- Output formats: `.json` and `.pdf`

## Requirements

- Python 3.11 or newer
- `uv`
- A machine capable of running the selected local LLM
- For RTX 50-series GPUs such as RTX 5070 Ti, this project resolves PyTorch from the official CUDA 12.8 wheel index
- For OCR fallback:
  - Tesseract OCR installed and available on `PATH`
  - Poppler installed and available on `PATH`, required by `pdf2image`

On Windows, `bitsandbytes` is not installed by default by this project. If CUDA is available but `bitsandbytes` is missing, the model loads without 4-bit quantization.

## Setup

```powershell
uv sync
```

Put secrets in `.env`:

```dotenv
HF_TOKEN=
```

`HF_TOKEN` is optional unless the selected Hugging Face model requires authentication.

## Configuration

Edit `config.toml` for internal runtime settings:

```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"
quantized = true

[generation]
max_new_tokens = 2500
chunk_size = 5000
chunk_overlap = 700

[ocr]
enabled = true
language = "eng"
dpi = 200
max_pages = 0
min_text_length = 300

[output]
directory = "output"
basename = "generated_quiz"
```

Use `max_pages = 0` to OCR all pages.

## Usage

Generate a 10-question single-choice quiz:

```powershell
uv run python main.py notes.pdf
```

## CLI Options

```text
main.py PDF
  --questions N
  --type single-choice|multiple-choice|open
  --language English
  --output-dir output
```
