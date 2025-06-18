import re
import os
import sys
import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Safe patterns (not personal)
SAFE_PATTERNS = [
    (re.compile(r"\bv\d+\.\d+\.\d+\b"), "[VERSION]"),
    (re.compile(r"\bPort\s+\d+\b", re.IGNORECASE), "[PORT]"),
    (re.compile(r"\bline\s+\d+\b", re.IGNORECASE), "[LINE]"),
    (re.compile(r"\b(?:ID|Ticket|Ref)[:#]?\s*\d+\b", re.IGNORECASE), "[ID]"),
    (re.compile(r"\b20\d{2}[-/]\d{2}[-/]\d{2}\b"), "[DATE]"),
]

# PII Redaction patterns
REDACTION_PATTERNS = [
    (re.compile(r"[\w\.-]+@[\w\.-]+\.\w{2,}"), "[EMAIL REDACTED]"),
    (re.compile(r"(?<!\w)(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}(?!\w)"), "[PHONE REDACTED]"),
    (re.compile(r"\b(?:\d{1,5}\s\w+(?:\s\w+)*\s(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr))\b", re.IGNORECASE), "[ADDRESS REDACTED]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CREDIT CARD REDACTED]"),
    (re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{2,4}\b"), "[POTENTIAL DOB REDACTED]"),
    (re.compile(r"\b00D\w{10,}\b"), "[TOKEN REDACTED]"),
]

# Form field patterns (structured full names)
FORM_FIELD_PATTERNS = [
    (re.compile(r"(First Name|Last Name|Contact Name):\s*[A-Z][a-z]+", re.IGNORECASE), r"\1: [PERSON REDACTED]"),
]

def mask_safe_patterns(text):
    for pattern, token in SAFE_PATTERNS:
        text = pattern.sub(token, text)
    return text

def redact_pii(text, log):
    text = mask_safe_patterns(text)
    for pattern, replacement in REDACTION_PATTERNS:
        for match in pattern.findall(text):
            log.append((pattern.pattern, match))
        text = pattern.sub(replacement, text)
    return text

def redact_form_fields(text):
    for pattern, replacement in FORM_FIELD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def redact_ner(text, log):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "ORG", "LOC"}:
            log.append((ent.label_, ent.text))
            text = text.replace(ent.text, f"[{ent.label_} REDACTED]")
    return text

def redact_text(text, log):
    if not isinstance(text, str):
        return text
    text = redact_pii(text, log)
    text = redact_form_fields(text)
    text = redact_ner(text, log)
    return text

def process_file(input_csv):
    base = os.path.splitext(os.path.basename(input_csv))[0]
    output_csv = f"{base}__redacted.csv"
    log_file = f"{base}__redaction_log.txt"
    log = []

    print(f"Processing: {input_csv}")
    df = pd.read_csv(input_csv)
    redacted_df = df.applymap(lambda x: redact_text(x, log))
    redacted_df.to_csv(output_csv, index=False)

    with open(log_file, "w", encoding="utf-8") as f:
        for pattern, match in log:
            f.write(f"{pattern} => {match}\n")

    print(f"Redacted: {output_csv}")
    print(f"Log: {log_file}\n")

def main():
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        files = [f for f in os.listdir('.') if f.endswith('.csv')]

    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()
