from collections import Counter
import hashlib
import ipaddress
import tempfile
import re
import string
import sys
from pathlib import Path

import pandas as pd
import phonenumbers
from bs4 import BeautifulSoup, Comment
from bs4.element import AttributeValueList
from docx import Document
from presidio_analyzer import (
    AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer,
    RecognizerRegistry, RecognizerResult,
)
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from wordfreq import zipf_frequency
from pdfminer.high_level import extract_text


class InternationalPhoneNumberRecognizer(PatternRecognizer):
    def __init__(self):
        """
        A recognizer specialized for US phone numbers.

        It detects:
        - +1XXXXXXXXXX (E.164 international US format)
        - (xxx) xxx-xxxx
        - xxx-xxx-xxxx
        - xxx xxx xxxx
        - xxxxxxxxxx

        And rejects:
        - +44... (UK)
        - +972... (Israel)
        - +49... (Germany)
        - short numeric sequences
        - ambiguous "20 25 55" style patterns
        """
        pattern = Pattern(
            name="international_us_number",
            regex=r"(?:\+1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})",
            score=0.7
        )
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=[pattern]
        )

    def validate_result(self, text: str) -> bool:
        """
        Only accept:
        - valid +1 E.164 numbers
        - valid US national-format numbers
        Reject everything else.
        """
        text = text.strip()

        if not any(c.isdigit() for c in text):
            return False

        # If number starts with "+1", parse as E.164 first
        if text.startswith("+1"):
            try:
                # Try to parse the number with no region bias (None)
                num = phonenumbers.parse(text, None)
                return phonenumbers.is_valid_number(num) and num.country_code == 1
            except phonenumbers.NumberParseException:
                # Just skip to next validation layer
                pass

        # National US parsing
        try:
            num_us = phonenumbers.parse(text, "US")
            if phonenumbers.is_valid_number(num_us):
                return True
        except phonenumbers.NumberParseException:
            # Not a valid US national format; try other parsing options
            pass

        # Try E.164 again with no region (for robustness), but accept ONLY US
        try:
            num_global = phonenumbers.parse(text, None)
            if phonenumbers.is_valid_number(num_global):
                return num_global.country_code == 1
        except phonenumbers.NumberParseException:
            return False

        return False


UNICODE_SPACE_MAP = {
    "\u00A0": " ",  # NBSP
    "\u202F": " ",  # NARROW NBSP
    "\u2007": " ",  # FIGURE SPACE
}
UNICODE_DASH_MAP = {
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2212": "-",  # MINUS SIGN
}


def _normalize_unicode_separators(s: str) -> str:
    # fast path: only do replaces if needed
    for u, r in UNICODE_SPACE_MAP.items():
        if u in s:
            s = s.replace(u, r)
    for u, r in UNICODE_DASH_MAP.items():
        if u in s:
            s = s.replace(u, r)
    return s


# === Presidio / redaction config ===
REDACT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "IBAN",
    "US_SSN",
    "TOKEN",
    # Can be added more as needed
]

ANALYZER_THRESHOLD = 0.6

#Anonymizer operators
ANONYMIZER_OPERATORS = {
    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL ADDRESS REDACTED]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE NUMBER REDACTED]"}),
    "PERSON": OperatorConfig("replace", {"new_value": "[PERSON REDACTED]"}),
    "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP ADDRESS REDACTED]"}),
    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT CARD REDACTED]"}),
    "IBAN": OperatorConfig("replace", {"new_value": "[IBAN REDACTED]"}),
    "US_SSN": OperatorConfig("replace", {"new_value": "[US SSN REDACTED]"}),
    "TOKEN": OperatorConfig("replace", {"new_value": "[TOKEN REDACTED]"}),
}

# === Configuration ===
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "."
OUTPUT_DIR = "redacted"
LOG_DIR = "logs"
SUPPORTED_EXTENSIONS = [".txt", ".csv", ".html", ".htm", ".docx", "pdf", ".log"]

RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
RE_TOKEN = re.compile(r"[a-zA-Z0-9]{10,}[.!][a-zA-Z0-9._-]{5,}[.!][a-zA-Z0-9._-]{5,}")
RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")
CONTEXT_POS = re.compile(r"(?i)\b(to|from|attn|cc|bcc|dear|regards|sincerely|signed|contact)\b")
CONTEXT_NEG = re.compile(r"(?i)\b(email|reason|status|owner|loading|workbook|user|name|value)\b")
_NAME_SEP = r"(?: |\u00A0|\u202F|\u2007)"
_FIRST_LAST = re.compile(rf"^[A-Z][a-z]+{_NAME_SEP}[A-Z][a-z]+$")
RE_FIRST_LAST = re.compile(rf"\b[A-Z][a-z]+{_NAME_SEP}[A-Z][a-z]+\b")
RE_E164_LAX = re.compile(r"\+\d[\d\-\s()]{6,}\d")

# Do not redact these suspicious "person" labels
FALSE_POSITIVE_PERSONS = {s.lower() for s in {
    "email", "reason", "loading", "workbook", "status", "owner", "assigned",
    "priority", "default", "total", "value", "name", "user", "users"
}}


# Quoted, ODBC brace-delimited, or unquoted field values.
VALUE = r'''"(?:\\.|""|[^"\\\r\n])*"|'(?:\\.|''|[^'\\\r\n])*'|\{(?:\}\}|[^}\r\n])*\}|[^\s;,<>"'\[\]{}]+'''


def fields(keys, allow_unquoted=True):
    # Also accepts quoted JSON keys.
    value = VALUE if allow_unquoted else VALUE.rsplit("|", 1)[0]
    return rf'''(?P<prefix>(?<![\w-])(?:{keys})["']?[ \t]*[:=][ \t]*)(?P<value>{value})'''


LOG_RULES = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("SECRET", fields(r"password|passwd|pwd|secret|client[_-]?secret|"
                          r"(?:access[_-]?|refresh[_-]?|session[_-]?)token|"
                          r"api[_-]?key|aws[_-]?secret[_-]?access[_-]?key")),
        ("SECRET", r'''(?P<prefix>\bauthorization["']?[ \t]*:[ \t]*["']?(?:bearer|basic)[ \t]+)(?P<value>[A-Za-z0-9._~+/=-]+)'''),
        ("SECRET", r"(?P<value>\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b)"),
        ("HOST", r'''(?P<prefix>\b[a-z][a-z0-9+.-]*://)(?P<value>[^/\s?#<>"']+)'''),
        ("URL_QUERY", r'''(?P<prefix>\b[a-z][a-z0-9+.-]*://[^?\s#<>"']*\?)(?P<value>[^\s<>"']+)'''),
        ("USER", fields(r"user(?:[ _-]?(?:name|id))?|uid|login(?:[ _-]?name)?")),
        ("PERSON", fields(r"full[ _-]?name|contact[ _-]?name", allow_unquoted=False)),
        ("PERSON", r'''(?P<prefix>\b(?:full[ _-]?name|contact[ _-]?name)["']?[ \t]*[:=][ \t]*)(?P<value>(?![\[{"\x27])[^\r\n;,]+?)(?=[ \t]+[\w-]+[ \t]*[:=]|[ \t]*(?:[;,]|$))'''),
        ("PHONE", r'''(?P<prefix>\b(?:phone(?:[ _-]?number)?|mobile|tel)["']?[ \t]*[:=][ \t]*["']?)(?P<value>\+?\(?\d[\d(). -]{5,}\d)'''),
        ("HOST", fields(r"host(?:[ _-]?name)?|server(?:[ _-]?name)?|data[ _-]?source")),
        ("HOST", rf"(?P<prefix>\brunning on host[ \t]+)(?P<value>{VALUE})"),
        ("DATABASE", fields(r"database(?:[ _-]?name)?|initial[ _-]?catalog")),
        ("LICENSEE", r"(?P<prefix>\bLicensed to[ \t]+)(?P<value>[^\r\n]+?)(?=,[ \t]*(?:term|evaluation|perpetual|subscription) license\b)"),
        ("BUCKET", r'''(?P<prefix>\bS3 bucket[ \t]+)(?P<value>"[^"\r\n]+"|'[^'\r\n]+')'''),
        ("USER", r'''(?P<prefix>\b[A-Z]:[\\/]Users[\\/]|/(?:home|Users)/)(?P<value>[^/\\\s"';<>]+)'''),
        ("EMAIL", r"(?P<value>[\w.!#$%&'*+/=?^\x60{|}~-]+@[\w.-]+\.[^\W\d_]{2,})"),
        ("US_SSN", r"(?P<value>(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-]))"),
    )
)
IP_CANDIDATE = re.compile(
    r"(?<![\w.:])(?:"
    r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?!\w|\.[0-9])"
    r"|(?:[0-9a-f]{0,4}:){2,}[0-9a-f:.]*(?:%[\w.-]+)?(?![\w:])"
    r")", re.IGNORECASE
)


LOG_PREFIX = re.compile(
    r"^\d+: \d{4}-\d\d-\d\dT\S+[ \t]+\[[^\]\r\n]+\][A-Z]:[ \t]*"
)
LOG_ENTITIES = sorted({f"LOG_{label}" for label, _ in LOG_RULES} | {"LOG_IP"})
LOG_ANALYZED_ENTITIES = [entity for entity in REDACT_ENTITIES if entity != "EMAIL_ADDRESS"] + LOG_ENTITIES


class LogFieldRecognizer(EntityRecognizer):
    """Return spans for sensitive values in support logs."""

    def __init__(self):
        super().__init__(supported_entities=LOG_ENTITIES, supported_language="en")

    def load(self):
        pass  # The patterns are compiled at module load.

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for label, rule in LOG_RULES:
            entity = f"LOG_{label}"
            if entity not in entities:
                continue
            for match in rule.finditer(text):
                start, end = match.span("value")
                value = text[start:end]
                if (value[0], value[-1]) in {('"', '"'), ("'", "'"), ("{", "}")}:
                    start += 1
                    end -= 1
                if start < end and not re.fullmatch(r"\[REDACTED_[A-Z_]+\]", text[start:end]):
                    results.append(RecognizerResult(entity, start, end, 1.0))

        if "LOG_IP" in entities:
            for match in IP_CANDIDATE.finditer(text):
                address = match[0].rstrip(".")
                try:
                    ipaddress.ip_address(address)
                except ValueError:
                    continue
                results.append(RecognizerResult("LOG_IP", match.start(), match.start() + len(address), 1.0))
        return results


LOG_OPERATORS = {
    **ANONYMIZER_OPERATORS,
    **{
        entity: OperatorConfig("replace", {"new_value": f"[REDACTED_{entity[4:]}]"})
        for entity in LOG_ENTITIES
    },
}


def process_log(source: Path, destination: Path):
    """Stream a UTF-8 log through the existing Presidio engines."""
    counts = Counter()
    temporary_path = None
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    try:
        with source.open(encoding="utf-8", newline="") as input_file:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=destination.parent,
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                for line in input_file:
                    if "\x00" in line:
                        raise ValueError("Expected a UTF-8 text log.")
                    prefix = LOG_PREFIX.match(line.lstrip("\ufeff"))
                    start = (1 if line.startswith("\ufeff") else 0) + (prefix.end() if prefix else 0)
                    end = len(line.rstrip("\r\n"))
                    message = line[start:end]
                    if message.strip():
                        results = ANALYZER.analyze(
                            text=message,
                            entities=LOG_ANALYZED_ENTITIES,
                            language="en",
                            score_threshold=ANALYZER_THRESHOLD,
                        )
                        result = ANONYMIZER.anonymize(
                            text=message, analyzer_results=results, operators=LOG_OPERATORS,
                        )
                        counts.update(item.entity_type.removeprefix("LOG_") for item in result.items)
                        message = result.text
                    output_file.write(line[:start] + message + line[end:])
        destination.hardlink_to(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return counts


def build_presidio():
    models = [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
        #{"lang_code": "he", "model_name": "he_core_news_lg"},  # for future usage
    ]
    nlp_engine = SpacyNlpEngine(models=models)
    try:
        nlp_engine.load()
    except Exception as ex:
        raise RuntimeError(
            "Failed to load spaCy model(s). Ensure `python -m spacy download en_core_web_lg` "
            "was run inside this virtualenv.\nOriginal error: " + str(ex)
        )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    token_recognizer = PatternRecognizer(
        name="TOKEN_RECOGNIZER",
        supported_entity="TOKEN",
        patterns=[Pattern(name="token", regex=RE_TOKEN.pattern, score=0.8)],
    )

    registry.add_recognizer(InternationalPhoneNumberRecognizer())
    registry.add_recognizer(token_recognizer)
    registry.add_recognizer(LogFieldRecognizer())

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


ANALYZER, ANONYMIZER = build_presidio()


def save_redacted_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def is_suspected_name(t: str) -> bool:
    t = t.strip().strip(string.punctuation)
    if not t:
        return False
    if t.lower() in FALSE_POSITIVE_PERSONS:
        return False
    #avoid obvious non-names
    if len(t.split()) > 3 or t.isupper() or t.isnumeric():
        return False
    if _FIRST_LAST.match(t):
        return True
    #common words filter (zipf>=4.5 are common)
    words = re.findall(r"[A-Za-z]+", t)
    if any(zipf_frequency(w.lower(), "en") >= 4.5 for w in words):
        return False
    return True


def should_redact_person(span_text: str, left_ctx: str, right_ctx: str) -> bool:
    if CONTEXT_NEG.search(left_ctx) or CONTEXT_NEG.search(right_ctx):
        return False
    if CONTEXT_POS.search(left_ctx) or CONTEXT_POS.search(right_ctx):
        return True
    return is_suspected_name(span_text)


def redact_phones_with_validator(text: str, log=None) -> str:
    """
    Redact phone numbers in local (IL) and international (+<cc>) formats.
    Tries region='IL' first for local numbers, then falls back to None for global.
    """
    # Normalize Unicode dashes → simple hyphen
    text_norm = text.replace("–", "-").replace("—", "-")

    out = []

    matches = []

    # Pass 1: local Israeli numbers (region='IL')
    for m in phonenumbers.PhoneNumberMatcher(
            text_norm, "IL", leniency=phonenumbers.Leniency.POSSIBLE
    ):
        if phonenumbers.is_valid_number(m.number):
            matches.append((m.start, m.end))

    # Pass 2: international +country code numbers (region=None)
    for m in phonenumbers.PhoneNumberMatcher(
            text_norm, None, leniency=phonenumbers.Leniency.POSSIBLE
    ):
        if phonenumbers.is_valid_number(m.number):
            matches.append((m.start, m.end))

    # Merge overlaps (rare but safe)
    matches = sorted(set(matches))
    last_end = 0
    for start, end in matches:
        if start < last_end:  # overlapping match, skip
            continue
        out.append(text_norm[last_end:start])
        span = text_norm[start:end]
        if log is not None:
            log.append(("PHONE_NUMBER", span))
        out.append("[PHONE REDACTED]")
        last_end = end
    out.append(text_norm[last_end:])

    return "".join(out)


def _fallback_redact_e164(text: str, log: list) -> str:
    out = []
    i = 0
    for m in RE_E164_LAX.finditer(text):
        start, end = m.start(), m.end()
        out.append(text[i:start])
        cand = m.group(0)
        try:
            num = phonenumbers.parse(cand, None)  # None: infer from +<cc>
            if phonenumbers.is_valid_number(num):
                log.append(("PHONE_NUMBER", cand))
                out.append("[PHONE REDACTED]")
            else:
                out.append(cand)
        except Exception:
            out.append(cand)
        i = end
    out.append(text[i:])
    return "".join(out)


def redact_text(text: str, log: list) -> str:
    def log_and_replace(pattern, replacement, label_name):
        matches = list(pattern.finditer(text))
        for match in matches:
            log.append((label_name, match.group()))
        return pattern.sub(replacement, text)

    text = _normalize_unicode_separators(text)

    text = log_and_replace(RE_EMAIL, "[EMAIL REDACTED]", "EMAIL_ADDRESS")
    text = log_and_replace(RE_TOKEN, "[TOKEN REDACTED]", "TOKEN")
    text = log_and_replace(RE_SSN, "[US SSN REDACTED]", "US_SSN")
    text = log_and_replace(RE_IBAN, "[IBAN REDACTED]", "IBAN")
    text = redact_phones_with_validator(text, log=log)
    text = _fallback_redact_e164(text, log)

    #run Presidio analyzer
    results = ANALYZER.analyze(
        text=text,
        entities=REDACT_ENTITIES,
        language="en",
        score_threshold=ANALYZER_THRESHOLD,
    )

    #anonymize with deterministic replacements
    anonymized = ANONYMIZER.anonymize(
        text=text,
        # analyzer_results=filtered,
        analyzer_results=results,
        operators=ANONYMIZER_OPERATORS
    ).text

    for r in results:
        span = text[r.start:r.end]
        log.append((r.entity_type, span))

    return anonymized


def process_txt(file_path_txt: Path, out_path: Path, log_path: Path):
    log = []
    with open(file_path_txt, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    redacted = redact_text(text, log)
    save_redacted_text(redacted, out_path)
    write_log(log, log_path)


def process_csv(file_path_csv: Path, out_path: Path, log_path: Path):
    def _redact_cell(x, log_list):
        if pd.isna(x):
            return x
        s = str(x).strip()
        return redact_text(s, log_list)

    log = []
    df = pd.read_csv(file_path_csv, dtype=str, encoding="utf-8", on_bad_lines="skip")
    for col in df.columns:
        df[col] = df[col].apply(lambda v: _redact_cell(v, log))
    df.to_csv(out_path, index=False)
    write_log(log, log_path)


def process_html(file_path_html: Path, out_path: Path, log_path: Path):
    log = []
    with open(file_path_html, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    #redact a visible text
    for node in soup.find_all(string=True):
        if node.parent.name in {"script", "style"} or isinstance(node, Comment):
            continue
        redacted = redact_text(str(node), log)
        if redacted != str(node):
            node.replace_with(soup.new_string(redacted))

    #redact inside element attributes
    pii_attrs = {"href", "title", "alt", "content", "data-email", "data-user"}
    for tag in soup.find_all(True):  # all element tags
        for attr, val in list(tag.attrs.items()):
            #list-valued attributes
            if isinstance(val, list):
                new_list, changed = [], False
                for v in val:
                    if isinstance(v, str):
                        red = redact_text(v, log)
                        if red != v:
                            changed = True
                        new_list.append(red)
                    else:
                        new_list.append(v)
                if changed:
                    if attr in {"class", "rel"}:
                        tag[attr] = AttributeValueList(new_list)
                    else:
                        tag[attr] = " ".join(new_list)
                continue
            #string attributes (filter to likely-PII attributes or any data-*)
            if isinstance(val, str) and (attr in pii_attrs or attr.startswith("data-")):
                red = redact_text(val, log)
                if red != val:
                    tag[attr] = red

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    write_log(log, log_path)


def process_docx(file_path_doc: Path, out_path: Path, log_path: Path):
    log = []
    doc = Document(str(file_path_doc))

    #redact regular paragraphs
    for para in doc.paragraphs:
        for run in para.runs:
            if run.text:
                run.text = redact_text(run.text, log)

    #redact text inside tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        if r.text:
                            r.text = redact_text(r.text, log)

    #redact headers and footers
    for section in doc.sections:
        #header
        for p in section.header.paragraphs:
            for r in p.runs:
                if r.text:
                    r.text = redact_text(r.text, log)
        #footer
        for p in section.footer.paragraphs:
            for r in p.runs:
                if r.text:
                    r.text = redact_text(r.text, log)

    doc.save(str(out_path))
    write_log(log, log_path)


def process_pdf_to_txt(file_path_pdf: Path, out_path: Path, log_path: Path):
    log = []
    text = extract_text(file_path_pdf)
    redacted = redact_text(text, log)
    out_path = out_path.with_suffix(".txt")
    save_redacted_text(redacted, out_path)
    write_log(log, log_path)


def write_log(log_entries, log_path):
    def _mask_for_log(x: str) -> str:
        h = hashlib.sha256(x.encode("utf-8")).hexdigest()[:8]
        #small preview for debugging only; must be removed if a policy forbids
        return f"<{x[:3]}…#{h}>"

    with open(log_path, "w", encoding="utf-8") as f:
        for label, value in log_entries:
            f.write(f"{label} => {_mask_for_log(value)}\n")


Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(exist_ok=True)

input_path = Path(INPUT_PATH)
files = []
if input_path.is_dir():
    for ext in SUPPORTED_EXTENSIONS:
        for candidate in input_path.rglob(f"*{ext}"):
            if ext == ".log" and Path(OUTPUT_DIR).resolve() in candidate.resolve().parents:
                continue
            files.append(candidate)
elif input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
    files = [input_path]
else:
    print("Unsupported file or directory.")
    sys.exit(1)

log_failed = False
for file_path in files:
    file_path = Path(file_path)
    name, ext = file_path.stem, file_path.suffix.lower()
    out_file = Path(OUTPUT_DIR) / f"{name}_redacted{ext}"
    log_file = Path(LOG_DIR) / f"{name}_log.txt"
    try:
        if ext == ".log":
            counts = process_log(file_path, out_file)
            print(", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
                  or "No sensitive values detected.")
        elif ext == ".txt":
            process_txt(file_path, out_file, log_file)
        elif ext == ".csv":
            process_csv(file_path, out_file, log_file)
        elif ext in [".html", ".htm"]:
            process_html(file_path, out_file, log_file)
        elif ext == ".docx":
            process_docx(file_path, out_file, log_file)
        elif ext == ".pdf":
            process_pdf_to_txt(file_path, out_file, log_file)
        print(f"Redacted: {file_path.name}")
    except Exception as e:
        if ext == ".log":
            log_failed = True
            if isinstance(e, UnicodeError):
                e = "Expected a UTF-8 text log."
        print(f"Error processing {file_path.name}: {e}")

if log_failed:
    sys.exit(1)
