"""Guardrail evaluation.

Rules run in the order the policy lists them, on the text segments a provider
adapter extracted from the request (or response) body. A rule can:

  block   refuse the request with 403 and an explanation
  redact  rewrite the offending span and let the request continue
  warn    record the finding, change nothing
  allow   record nothing (useful to switch a rule off without deleting it)

Every rule is pure: text in, findings + text out. Nothing here does IO, so a
policy can be exercised against a corpus offline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..pii.patterns import DETECTORS
from ..pii.redact import scan_text
from ..policy import Rule
from . import injection
from .base import Finding, Outcome

REDACTION_MARK = "[BLOCKED]"

# Labels a "secrets" rule cares about -- credentials, not personal data.
SECRET_LABELS = ("KEY", "TOKEN", "SECRET", "URL")


def _word_regex(word: str) -> "re.Pattern":
    w = word.strip()
    if not w:
        return re.compile(r"(?!x)x")
    if re.match(r"^[\w\s'-]+$", w):
        return re.compile(r"(?<!\w)" + re.escape(w).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
    return re.compile(re.escape(w), re.I)


_TOPIC_CACHE: Dict[int, Dict[str, List["re.Pattern"]]] = {}


def _topic_patterns(rule: Rule) -> Dict[str, List["re.Pattern"]]:
    cached = _TOPIC_CACHE.get(id(rule))
    if cached is None:
        cached = {name: [_word_regex(w) for w in words]
                  for name, words in rule.topics.items()}
        _TOPIC_CACHE[id(rule)] = cached
    return cached


def _apply_spans(text: str, spans: Sequence[Tuple[int, int]], mark: str = REDACTION_MARK) -> str:
    if not spans:
        return text
    merged: List[List[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: List[str] = []
    last = 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(mark)
        last = e
    out.append(text[last:])
    return "".join(out)


def _secret_spans(text: str) -> Tuple[List[Tuple[int, int]], List[str]]:
    spans: List[Tuple[int, int]] = []
    labels: List[str] = []
    for det in DETECTORS:
        if det.label not in SECRET_LABELS:
            continue
        for m in det.re.finditer(text):
            captured = m.group(det.group) if det.group else m.group(0)
            if not captured or (det.validate and not det.validate(captured)):
                continue
            start = m.start(det.group) if det.group else m.start()
            spans.append((start, start + len(captured)))
            if det.label not in labels:
                labels.append(det.label)
    return spans, labels


def _run_rule(rule: Rule, text: str, ctx: Dict[str, Any]) -> Tuple[Optional[float], List[str], List[Tuple[int, int]]]:
    """Returns (score or None when the rule did not fire, detail, spans)."""
    if rule.type == "injection":
        s, hits, spans = injection.score(text)
        return (s, hits, spans) if s >= rule.threshold else (None, [], [])

    if rule.type == "topics":
        detail: List[str] = []
        spans: List[Tuple[int, int]] = []
        for name, patterns in _topic_patterns(rule).items():
            for pat in patterns:
                for m in pat.finditer(text):
                    spans.append((m.start(), m.end()))
                    if name not in detail:
                        detail.append(name)
        return (1.0, detail, spans) if detail else (None, [], [])

    if rule.type == "regex":
        spans = []
        for pat in rule.patterns:
            for m in pat.finditer(text):
                spans.append((m.start(), m.end()))
        return (1.0, [rule.id], spans) if spans else (None, [], [])

    if rule.type == "secrets":
        spans, labels = _secret_spans(text)
        return (1.0, labels, spans) if spans else (None, [], [])

    if rule.type == "pii_leak":
        labels = scan_text(text, strict=bool(ctx.get("strict")))
        return (1.0, labels, []) if labels else (None, [], [])

    if rule.type == "unicode":
        n = injection.invisible_count(text)
        if not n:
            return None, [], []
        spans = [(m.start(), m.end()) for m in injection.INVISIBLE_RE.finditer(text)]
        return 1.0, ["invisible_characters:%d" % n], spans

    if rule.type == "max_length":
        total = int(ctx.get("total_chars", len(text)))
        if total <= rule.max_chars:
            return None, [], []
        return 1.0, ["%d chars > %d" % (total, rule.max_chars)], []

    if rule.type == "model_allowlist":
        model = str(ctx.get("model") or "")
        if not model or any(re.fullmatch(p.replace("*", ".*"), model) for p in rule.models):
            return None, [], []
        return 1.0, [model], []

    return None, [], []


def evaluate(rules: Sequence[Rule], segments: Sequence[str], direction: str = "input",
             ctx: Optional[Dict[str, Any]] = None) -> Outcome:
    ctx = dict(ctx or {})
    out = Outcome(segments=list(segments))
    ctx.setdefault("total_chars", sum(len(s) for s in out.segments))

    for rule in rules:
        if rule.action == "allow":
            continue
        # Whole-request rules see the request once; text rules see each segment.
        whole = rule.type in ("max_length", "model_allowlist")
        targets = [(-1, "")] if whole else list(enumerate(out.segments))
        for idx, text in targets:
            if not whole and not text:
                continue
            score, detail, spans = _run_rule(rule, text, ctx)
            if score is None:
                continue
            finding = Finding(rule_id=rule.id, type=rule.type, action=rule.action,
                              score=score, detail=detail, segment=idx,
                              direction=direction,
                              message=rule.message or _default_message(rule, detail))
            out.findings.append(finding)
            if rule.action == "block":
                out.blocked = finding
                return out
            if rule.action == "redact" and spans and idx >= 0:
                out.segments[idx] = _apply_spans(text, spans,
                                                 "" if rule.type == "unicode" else REDACTION_MARK)
                out.changed = True
    return out


def _default_message(rule: Rule, detail: Sequence[str]) -> str:
    what = ", ".join(detail[:4]) or rule.id
    return {
        "injection": "Prompt injection signals detected (%s)." % what,
        "topics": "Request matched a restricted topic (%s)." % what,
        "regex": "Request matched a restricted pattern (%s)." % rule.id,
        "secrets": "Credential-shaped content detected (%s)." % what,
        "pii_leak": "Personal data detected in the model response (%s)." % what,
        "unicode": "Invisible control characters detected (%s)." % what,
        "max_length": "Request exceeds the configured size limit (%s)." % what,
        "model_allowlist": "Model %s is not permitted by this policy." % what,
    }.get(rule.type, "Blocked by rule %s." % rule.id)
