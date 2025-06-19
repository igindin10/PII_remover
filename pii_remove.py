import re
import string
import sys
from pathlib import Path

import nltk
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from docx import Document
from nltk.corpus import words as nltk_words

nltk.download("words")
english_words = set(nltk_words.words())

# === Configuration ===
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "."
OUTPUT_DIR = "redacted"
LOG_DIR = "logs"
SUPPORTED_EXTENSIONS = [".txt", ".csv", ".html", ".htm", ".docx"]

# Updated regex
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
RE_TOKEN = re.compile(r"[a-zA-Z0-9]{10,}[.!][a-zA-Z0-9._-]{5,}[.!][a-zA-Z0-9._-]{5,}")
SAFE_ENTITY_LABELS = {"PERSON"}

# Do not redact these suspicious "person" labels
FALSE_POSITIVE_PERSONS = {"email", "reason", "loading", "workbook"}

nlp = spacy.load("en_core_web_sm")

def save_redacted_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def is_suspected_name(text):
    text = text.strip().strip(string.punctuation)
    if not text:
        return False
    if text.lower() in FALSE_POSITIVE_PERSONS:
        return False
    return (
        len(text.split()) <= 3
        and not text.isupper()
        and not text.isnumeric()
        and all(w.lower() not in english_words for w in text.split())
    )

def redact_text(text, log):
    def log_and_replace(pattern, replacement, label_name):
        matches = list(pattern.finditer(text))
        for match in matches:
            log.append((label_name, match.group()))
        return pattern.sub(replacement, text)

    text = log_and_replace(RE_EMAIL, "[EMAIL REDACTED]", "EMAIL")
    text = log_and_replace(RE_TOKEN, "[TOKEN REDACTED]", "TOKEN")

    doc = nlp(text)
    redacted = text
    offset = 0
    for ent in doc.ents:
        if ent.label_ in SAFE_ENTITY_LABELS and is_suspected_name(ent.text):
            start = ent.start_char + offset
            end = ent.end_char + offset
            replacement = f"[{ent.label_} REDACTED]"
            redacted = redacted[:start] + replacement + redacted[end:]
            offset += len(replacement) - (end - start)
            log.append((ent.label_, ent.text))
    return redacted

def process_txt(file_path, out_path, log_path):
    log = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    redacted = redact_text(text, log)
    save_redacted_text(redacted, out_path)
    write_log(log, log_path)

def process_csv(file_path, out_path, log_path):
    log = []
    df = pd.read_csv(file_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    for col in df.columns:
        df[col] = df[col].astype(str).apply(lambda x: redact_text(x, log))
    df.to_csv(out_path, index=False)
    write_log(log, log_path)

def process_html(file_path, out_path, log_path):
    log = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup.find_all(text=True):
        redacted = redact_text(tag, log)
        tag.replace_with(redacted)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    write_log(log, log_path)

def process_docx(file_path, out_path, log_path):
    log = []
    doc = Document(file_path)
    for para in doc.paragraphs:
        para.text = redact_text(para.text, log)
    doc.save(out_path)
    write_log(log, log_path)

def write_log(log_entries, log_path):
    with open(log_path, "w", encoding="utf-8") as log_file:
        for label, value in log_entries:
            log_file.write(f"{label} => {value}\n")

Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(exist_ok=True)

input_path = Path(INPUT_PATH)
files = []
if input_path.is_dir():
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_path.rglob(f"*{ext}"))
elif input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
    files = [input_path]
else:
    print("Unsupported file or directory.")
    sys.exit(1)

for file_path in files:
    file_path = Path(file_path)
    name, ext = file_path.stem, file_path.suffix.lower()
    out_file = Path(OUTPUT_DIR) / f"{name}_redacted{ext}"
    log_file = Path(LOG_DIR) / f"{name}_log.txt"
    try:
        if ext == ".txt":
            process_txt(file_path, out_file, log_file)
        elif ext == ".csv":
            process_csv(file_path, out_file, log_file)
        elif ext in [".html", ".htm"]:
            process_html(file_path, out_file, log_file)
        elif ext == ".docx":
            process_docx(file_path, out_file, log_file)
        print(f"Redacted: {file_path.name}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
