"""PII detectors used by the redaction engine (``redact.py``).

Every entry is applied IN ORDER to the text, so the most specific patterns come
first: a card number must be claimed before the phone detector can eat it, an
email before the phone detector sees the digits inside it.

``strict`` detectors only run when strict mode is on -- they trade false
positives for coverage (a bare 10-digit number is a phone number in a chat
message and an ID in a stack trace; we cannot tell).

A detector never decides policy. It reports a span; the engine replaces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Pattern


@dataclass(frozen=True)
class Detector:
    label: str
    re: Pattern
    group: int = 0
    validate: Optional[Callable[[str], bool]] = None
    strict: bool = False
    # Trims a matched span instead of rejecting it outright. The result must be
    # a suffix of the input, or None to drop the match.
    refine: Optional[Callable[[str], Optional[str]]] = None


# -- Validators --------------------------------------------------------------

def digits_of(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_valid(digits: str) -> bool:
    """Keeps 16-digit build numbers and random digit runs from being masked as
    payment cards. Real card numbers always satisfy it."""
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    double = False
    for ch in reversed(digits):
        if not ch.isdigit():
            return False
        d = int(ch)
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    return total % 10 == 0


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """The checksum Aadhaar (Indian national ID) uses. Without it every 12-digit
    number in a log line would be masked."""
    if len(digits) != 12 or digits[0] in "01":
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def is_public_ipv4(ip: str) -> bool:
    """Private / loopback / link-local addresses are not personal data -- masking
    them would break every networking question an assistant is asked."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        p = [int(x) for x in parts]
    except ValueError:
        return False
    if any(n < 0 or n > 255 for n in p):
        return False
    if p[0] in (0, 10, 127):
        return False
    if p[0] == 192 and p[1] == 168:
        return False
    if p[0] == 172 and 16 <= p[1] <= 31:
        return False
    if p[0] == 169 and p[1] == 254:
        return False
    return p[0] < 224


# Words that look like a name to a capitalisation heuristic but never are.
NAME_STOPWORDS = frozenset("""
the this that these those a an and or but if then else
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
i you he she we they it my your his her our their
react angular vue node java spring boot python typescript javascript kafka redis docker
kubernetes terraform postgres postgresql mongodb sqlite aws gcp azure google apple
microsoft amazon meta openai anthropic gemini claude chatgpt github gitlab
api http https json html css sql rest graphql grpc jwt
senior software engineer developer manager director intern team company client customer
user system service server
mac macos windows linux chrome safari firefox electron vite compose helm nginx
jenkins lambda kinesis athena bedrock vertex copilot cursor
hello hi hey thanks thank please sorry yes no ok okay
mail email emailed contact write writing send sent reply cc bcc to from via at on see
ping ask forward subject re
good morning afternoon evening night today tomorrow yesterday
""".split())


def _norm_word(w: str) -> str:
    return w.lower().strip(".'’-")


def _name_words_ok(m: str) -> bool:
    return bool(m) and all(_norm_word(w) not in NAME_STOPWORDS for w in m.split())


def _name_refine(m: str) -> Optional[str]:
    """"Email Jane Smith <...>" -- the capitalised verb in front of a name is not
    part of it. Drop leading stopwords rather than losing the whole span."""
    s = m
    while s.split():
        first = s.split()[0]
        if _norm_word(first) not in NAME_STOPWORDS:
            break
        s = s.lstrip()[len(first):].lstrip()
    if not s or not s[0].isupper():
        return None
    return s if _name_words_ok(s) else None


# Version strings ("1.2.3.4") and dotted quads inside longer numbers are the
# usual false positives -- require a clean boundary.
IPV4_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")

# Two or three capitalised words in a row, none a stopword, not sentence-initial.
# Behind strict mode: this fires on "Spring Boot Actuator" as readily as a person.
NAME_HEURISTIC_RE = re.compile(
    r"(?<![.!?]\s)(?<![A-Za-z'’-])([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z'’.-]{1,15}){1,2})(?![A-Za-z'’-])"
)

NAME_HEURISTIC = Detector(
    label="NAME", re=NAME_HEURISTIC_RE, group=1, refine=_name_refine, strict=True
)

DETECTORS = (
    Detector("KEY", re.compile(
        r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----")),
    Detector("TOKEN", re.compile(  # JWT
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b")),
    Detector("SECRET", re.compile(  # provider-shaped API keys
        r"\b(?:AIza[0-9A-Za-z_-]{30,40}|sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprse]-[A-Za-z0-9-]{10,}"
        r"|glpat-[A-Za-z0-9_-]{16,}|ih_[A-Za-z0-9]{24,})\b")),
    Detector("URL", re.compile(  # credentials embedded in a URL
        r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@\S+", re.I)),
    Detector("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")),
    # A name sitting right in front of an address -- "Jane Smith <j@x.com>", the
    # shape of every From: header -- is that person's name. Runs after EMAIL, so
    # it matches the placeholder and needs no strict-mode guesswork.
    Detector("NAME", re.compile(
        r"(?<![\w@.'’-])([A-Z][\w'’.-]*(?:[ ]+[A-Z][\w'’.-]*){0,2})[\"'’]?[ ]*(?:[<(][ ]*)?(?=\[EMAIL_\d+\])"),
        group=1, refine=_name_refine),
    # The account name in a home-directory path. Only the user segment goes, so
    # "/Users/[USER_1]/src" still reads as a path.
    Detector("USER", re.compile(
        r"(?:/Users/|/home/|\\Users\\)([A-Za-z0-9._-]{2,32})(?=[/\\ '\")\]:,;]|$)"),
        group=1,
        validate=lambda m: m.lower() not in
        ("shared", "public", "guest", "root", "admin", "local", "user")),
    Detector("CARD", re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
             validate=lambda m: luhn_valid(digits_of(m))),
    Detector("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}[ ]?[A-Z0-9]{1,4}\b"),
             validate=lambda m: 15 <= len(re.sub(r"\s", "", m)) <= 34),
    Detector("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    Detector("NATIONAL_ID", re.compile(r"(?<![\d-])\d{4}[ -]?\d{4}[ -]?\d{4}(?![\d-])"),
             validate=lambda m: verhoeff_valid(digits_of(m))),           # Aadhaar
    Detector("TAX_ID", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),            # Indian PAN
    Detector("PASSPORT", re.compile(
        r"\b(?:passport(?:\s*(?:no|number|#))?\s*[:#]?\s*)([A-Z]\d{7})\b", re.I), group=1),
    # International form: a leading + plus 8-15 digits is a phone number, nothing else.
    Detector("PHONE", re.compile(r"(?<![\w.])\+\d[\d\s().-]{7,18}\d(?![\w.])"),
             validate=lambda m: 8 <= len(digits_of(m)) <= 15),
    # Separated national forms: 98765 43210, 555-867-5309, (555) 867 5309.
    Detector("PHONE", re.compile(
        r"(?<![\w.+-])(?:\(\d{3}\)[ .-]?\d{3}[ .-]?\d{4}|\d{3}[ .-]\d{3}[ .-]\d{4}"
        r"|\d{5}[ .-]\d{5}|\d{4}[ .-]\d{3}[ .-]\d{3})(?![\w.-])"),
        validate=lambda m: len(digits_of(m)) == 10 and not m.lstrip("(").startswith("0")),
    # Bare digit runs only when something nearby says "phone".
    Detector("PHONE", re.compile(
        r"\b(?:phone|tel(?:ephone)?|mobile|cell|whatsapp|contact(?:\s*(?:no|number))?"
        r"|call\s*(?:me\s*(?:at|on))?)\b\s*(?:number|no|#)?\s*[:#-]?\s*(\+?\d[\d\s().-]{7,18}\d)", re.I),
        group=1),
    Detector("IP", IPV4_RE, group=1, validate=is_public_ipv4),
    Detector("MAC", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")),
    Detector("DOB", re.compile(
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|born(?:\s+on)?)\b\s*[:-]?\s*"
        r"([0-3]?\d[/.-][0-1]?\d[/.-](?:19|20)\d{2}|(?:19|20)\d{2}[/.-][0-1]?\d[/.-][0-3]?\d"
        r"|[A-Z][a-z]+\s+[0-3]?\d,?\s+(?:19|20)\d{2})", re.I), group=1),
    Detector("ADDRESS", re.compile(  # number + words + a street-type word
        r"\b\d{1,5}[A-Za-z]?(?:[ ,]+(?:[A-Z][\w'.-]*|[a-z]{1,4}))*[ ,]+"
        r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|court|ct|place|pl"
        r"|terrace|nagar|marg|colony|sector|block|apartments?|apt|flat|suite|ste|floor)\b\.?"
        r"(?:[ ,]+(?:apt|unit|suite|ste|flat|\#)\s*[\w-]+)?", re.I)),
    Detector("POSTCODE", re.compile(
        r"\b(?:zip|postal|pin)\s*(?:code)?\s*[:#-]?\s*"
        r"(\d{5}(?:-\d{4})?|\d{6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.I), group=1),
    Detector("GPS", re.compile(
        r"(?<![\w.])-?(?:[0-8]?\d|90)\.\d{4,}\s*,\s*-?(?:1[0-7]\d|[0-9]?\d|180)\.\d{4,}(?![\w.])")),
    Detector("HANDLE", re.compile(r"(?<![\w@/])@[A-Za-z][A-Za-z0-9_.]{2,29}\b"), strict=True),
)

ALL_LABELS = tuple(sorted({d.label for d in DETECTORS} | {"NAME"}))
