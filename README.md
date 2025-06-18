#  PII Redactor Tool

This tool scans and redacts Personally Identifiable Information (PII) from CSV files. It uses a combination of regular expressions and spaCy's Named Entity Recognition (NER) to clean sensitive content from customer support data.

##  Features

-  Detects and redacts:
  - Emails
  - Phone numbers
  - Addresses
  - Tokens & IDs
  - Names and organizations (via spaCy NER)
-  Keeps a redaction log for review
-  Supports:
  - Single file mode: `input.csv`
  - Batch mode: processes all `.csv` files in the current folder

##  Quick Start

### 1. Clone or download the project

```bash
git clone https://github.com/igindin10/PII_remover.git
cd PII_remover
```

Or unzip the provided archive.

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the spaCy model

```bash
python -m spacy link en_core_web_sm en_core_web_sm
```

> This step ensures spaCy can load the model by name.

##  How to Use

### Option 1: Redact a single CSV file

```bash
python pii_remove.py yourfile.csv
```

### Option 2: Redact all `.csv` files in the current directory

```bash
python pii_remove.py
```

##  Output

For each input CSV, you'll get:

-  `yourfile__redacted.csv` — cleaned version  
-  `yourfile__redaction_log.txt` — what was redacted

##  Notes

- This version uses `en_core_web_sm` — fast and good for demo/Hackathon purposes.
- For more accurate redaction, consider upgrading to `en_core_web_trf` (requires PyTorch).
- The tool assumes well-formed CSV files encoded in UTF-8.

## License

This project is intended for internal demo/hackathon use only. Not production-hardened.

## Authors

- Maintainer: Igor Gindin
