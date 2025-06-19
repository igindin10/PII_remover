# PII Redactor Tool

This tool scans and redacts Personally Identifiable Information (PII) from structured and unstructured documents using NLP and Regex.

## Supports:
- CSV
- Plain text
- Word documents (`.docx`)
- HTML files

## Features

- Detects and redacts:
  - Emails
  - API tokens / access keys
  - Named Entities: PERSON, ORG, GPE (via spaCy)
- Preserves formatting (in `.docx` and `.html`)
- Uses whitelist-based filtering (to avoid false positives like `Qlik Sense`)
- Redaction log per file (shows what was removed and why)

## Supported Input Modes

- Redact a single file:
  ```bash
  python pii_remove.py myfile.csv
  ```

- Redact all supported files in the current folder:
  ```bash
  python pii_remove.py
  ```

- Redact a specific folder:
  ```bash
  python pii_remove.py /path/to/mydata
  ```

## Output

For every input file:
- `myfile__redacted.csv|.docx|.html|.txt`
- `myfile__redaction_log.txt` — includes entity types and redacted content

## Technology Used

- `spaCy` – Named Entity Recognition
- `nltk` – English vocabulary filtering
- `re` – Email, token pattern detection
- `python-docx` – Modify `.docx` files while preserving layout
- `BeautifulSoup` – Clean HTML while keeping structure
- `pandas` – Parse `.csv` efficiently

## Setup

```bash
# Clone repo
git clone https://github.com/igindin10/PII_remover.git
cd PII_remover

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy model
python -m spacy download en_core_web_sm
```

## Notes

- Model: `en_core_web_sm` (can be upgraded to `en_core_web_trf` for production)
- Not production-hardened. Designed for Hackathon/demos.
- False positives minimized with tech-term filtering logic.

## Authors

Maintainer: Igor Gindin

## License

For internal hackathon/demo purposes only.
