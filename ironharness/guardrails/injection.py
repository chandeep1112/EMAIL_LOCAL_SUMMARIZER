"""Prompt-injection and jailbreak signals.

A weighted signal table, not a model: it runs in microseconds, needs no network
call, and -- crucially for a security product -- every decision it makes can be
explained to a customer by naming the signal that fired.

Signals are deliberately conservative. The cost of a false positive here is a
blocked customer request, so anything ambiguous ("as a language model") is left
out. Operators who want more aggression lower the rule's threshold; operators
who want an LLM-judge second opinion add one in front of this.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# (signal id, weight, pattern). Weight is the score a single hit contributes.
SIGNALS: Tuple[Tuple[str, float, "re.Pattern"], ...] = (
    ("instruction_override", 0.90, re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}?\b"
        r"(?:previous|prior|above|earlier|initial|original|all)\b[^.\n]{0,30}?"
        r"\b(?:instruction|prompt|rule|direction|context|message)s?\b", re.I)),
    ("instruction_override", 0.90, re.compile(
        r"\b(?:ignore|disregard)\b[^.\n]{0,20}\b(?:everything|all)\b[^.\n]{0,20}\b(?:above|before|said)\b", re.I)),
    ("system_prompt_exfil", 0.90, re.compile(
        r"\b(?:reveal|show|print|repeat|output|display|tell\s+me|what\s+(?:is|are|was))\b[^.\n]{0,30}?"
        r"\b(?:your|the|original)\b[^.\n]{0,20}?\b(?:system\s*prompt|initial\s*instructions?|"
        r"instructions?|rules?|guidelines?|configuration)\b", re.I)),
    ("system_prompt_exfil", 0.75, re.compile(
        r"\brepeat\b[^.\n]{0,25}\b(?:the\s+)?(?:text|words|everything|content)\b[^.\n]{0,15}\babove\b", re.I)),
    ("role_hijack", 0.70, re.compile(
        r"(?:<\|im_(?:start|end)\|>|\[/?INST\]|<<SYS>>|\bA:\s*###|^\s*###\s*(?:system|instruction)|"
        r"^\s*(?:system|assistant)\s*:\s)", re.I | re.M)),
    ("role_hijack", 0.65, re.compile(
        r"\byou\s+are\s+(?:now|no\s+longer)\b|\bfrom\s+now\s+on,?\s+you\b|"
        r"\bnew\s+(?:system\s+)?(?:instructions?|persona|role)\s*:", re.I)),
    ("jailbreak_persona", 0.70, re.compile(
        r"\b(?:DAN|do\s+anything\s+now|developer\s+mode|god\s*mode|jailbr(?:eak|oken)|"
        r"unfiltered\s+(?:mode|response)|without\s+(?:any\s+)?(?:restrictions?|filters?|guardrails?)|"
        r"no\s+longer\s+bound\s+by)\b", re.I)),
    ("jailbreak_persona", 0.55, re.compile(
        r"\bpretend\s+(?:that\s+)?you\s+(?:are|have\s+no|don'?t\s+have)\b|"
        r"\bact\s+as\s+(?:if\s+you\s+(?:are|have)|an?\s+unrestricted)\b|"
        r"\bhypothetically,?\s+if\s+you\s+(?:could|had\s+no)\b", re.I)),
    ("policy_pressure", 0.45, re.compile(
        r"\b(?:this\s+is\s+(?:just\s+)?(?:a\s+test|fiction|for\s+research)|"
        r"you\s+(?:have|are\s+given)\s+permission|the\s+(?:admin|developer|owner)\s+(?:says|allows)|"
        r"i\s+am\s+(?:the\s+)?(?:your\s+)?(?:developer|admin|creator))\b", re.I)),
    ("tool_abuse", 0.55, re.compile(
        r"\b(?:curl|wget)\b[^\n]{0,80}\|\s*(?:ba)?sh\b|\brm\s+-rf\s+/|"
        r"\b(?:cat|read)\s+(?:~/)?\.(?:env|aws/credentials|ssh/id_[a-z0-9]+)\b|"
        r"\bexfiltrat|\bsend\s+(?:the\s+)?(?:conversation|history|secrets?|keys?)\s+to\b", re.I)),
    ("markdown_exfil", 0.60, re.compile(
        r"!\[[^\]]*\]\(\s*https?://[^)\s]{4,}[?&/][^)\s]*(?:\{\{?|\$\{|%s|<[a-z_]+>)", re.I)),
    ("encoded_payload", 0.35, re.compile(
        r"\b(?:base64|rot13|atob|fromCharCode)\b[^\n]{0,40}\bdecode|"
        r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{240,}={0,2}(?![A-Za-z0-9+/])")),
)

# Characters that carry instructions no human reviewer can see: zero-width
# joiners, bidi overrides, and the Unicode tag block used for "invisible prompt"
# attacks.
INVISIBLE_RE = re.compile(
    "[​-‏‪-‮⁦-⁩﻿\U000e0000-\U000e007f]")


def strip_invisible(text: str) -> Tuple[str, int]:
    """Remove invisible control characters and normalise confusables away."""
    cleaned, n = INVISIBLE_RE.subn("", text)
    return cleaned, n


def invisible_count(text: str) -> int:
    return len(INVISIBLE_RE.findall(text or ""))


def score(text: str) -> Tuple[float, List[str], List[Tuple[int, int]]]:
    """Return (score 0-1, signal ids that fired, spans that matched).

    The score is the strongest single signal plus a small bonus per additional
    distinct signal -- two weak independent hints are worth more than one.
    """
    if not text:
        return 0.0, [], []
    # NFKC folds the fullwidth/mathematical lookalikes used to slip past regexes.
    probe = unicodedata.normalize("NFKC", text)
    hits: List[str] = []
    spans: List[Tuple[int, int]] = []
    best = 0.0
    for sid, weight, pattern in SIGNALS:
        m = pattern.search(probe)
        if not m:
            continue
        if sid not in hits:
            hits.append(sid)
        best = max(best, weight)
        for mm in pattern.finditer(probe):
            spans.append((mm.start(), mm.end()))
    if not hits:
        n_inv = invisible_count(text)
        if n_inv >= 4:
            return 0.8, ["invisible_characters"], []
        return 0.0, [], []
    if invisible_count(text) >= 1:
        hits.append("invisible_characters")
    total = min(1.0, best + 0.08 * (len(hits) - 1))
    return round(total, 3), hits, spans
