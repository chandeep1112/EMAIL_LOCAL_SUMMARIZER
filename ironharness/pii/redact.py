"""Reversible PII redaction.

The model never sees personal data: every detected span is swapped for a stable
placeholder ("[EMAIL_1]") before the request leaves the gateway, and the
placeholder is swapped back for the real value in the reply before it reaches
the caller. The map between the two -- the vault -- lives in the gateway process
only, in memory, and is never persisted or forwarded.

Stability matters as much as masking: the same email must get the same token for
a whole conversation, or the model loses track of who is who ("reply to
[EMAIL_1], not [EMAIL_2]").

Four modes:
  reversible  [EMAIL_1]        vault restores the real value in the response
  mask        [EMAIL]          irreversible, nothing to leak back
  hash        [EMAIL_9f3a12]   irreversible but stable across requests/processes
  off         no redaction

This module is dependency-free so it can be tested and vendored on its own.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .patterns import DETECTORS, NAME_HEURISTIC, Detector

TOKEN_RE = re.compile(r"\[([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*?)_([0-9]+|[0-9a-f]{6,})\]")
MAX_VAULT_ENTRIES = 2000

MODES = ("reversible", "mask", "hash", "off")


class Vault:
    """Placeholder <-> value map for one conversation."""

    __slots__ = ("_by_value", "_by_token", "_counters", "_hits", "_lock",
                 "mode", "_secret", "created_at", "last_used")

    def __init__(self, mode: str = "reversible", secret: bytes = b"") -> None:
        self.mode = mode if mode in MODES else "reversible"
        self._secret = secret
        self._by_value: Dict[str, str] = {}
        self._by_token: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        self._hits: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.created_at = time.time()
        self.last_used = self.created_at

    def token_for(self, label: str, value: str) -> str:
        key = label + ":" + value.lower()
        with self._lock:
            self._hits[label] = self._hits.get(label, 0) + 1
            existing = self._by_value.get(key)
            if existing:
                return existing
            if self.mode == "mask":
                token = "[%s]" % label
            elif self.mode == "hash":
                digest = hmac.new(self._secret or b"ironharness",
                                  key.encode("utf-8"), hashlib.sha256).hexdigest()[:8]
                token = "[%s_%s]" % (label, digest)
            elif len(self._by_token) >= MAX_VAULT_ENTRIES:
                # A full vault stops minting rather than evicting: evicting would
                # silently break restore() for a conversation still in flight.
                # Anything past the cap is masked irreversibly -- the safe
                # direction to fail in.
                return "[%s]" % label
            else:
                n = self._counters.get(label, 0) + 1
                self._counters[label] = n
                token = "[%s_%d]" % (label, n)
            self._by_value[key] = token
            if self.mode != "mask":
                self._by_token[token] = value
            return token

    def restore(self, text: Optional[str]) -> Optional[str]:
        if not text or self.mode != "reversible" or not self._by_token:
            return text
        return TOKEN_RE.sub(lambda m: self._by_token.get(m.group(0), m.group(0)), str(text))

    def counts(self) -> Dict[str, int]:
        return dict(self._hits)

    def total(self) -> int:
        return sum(self._hits.values())

    @property
    def size(self) -> int:
        return len(self._by_token)

    def clear(self) -> None:
        with self._lock:
            self._by_value.clear()
            self._by_token.clear()
            self._counters.clear()
            self._hits.clear()


class VaultStore:
    """Keeps a vault alive across the turns of one conversation.

    Clients opt in by sending the same ``X-IronHarness-Session`` header on every
    turn; without it each request gets a throwaway vault (still internally
    consistent, just not stable across turns).
    """

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 5000) -> None:
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._vaults: Dict[str, Vault] = {}
        self._lock = threading.Lock()

    def get(self, session_id: Optional[str], mode: str, secret: bytes = b"") -> Vault:
        if not session_id:
            return Vault(mode, secret)
        now = time.time()
        with self._lock:
            self._evict(now)
            v = self._vaults.get(session_id)
            if v is None or v.mode != mode:
                v = Vault(mode, secret)
                self._vaults[session_id] = v
            v.last_used = now
            return v

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._vaults.pop(session_id, None) is not None

    def _evict(self, now: float) -> None:
        stale = [k for k, v in self._vaults.items() if now - v.last_used > self.ttl]
        for k in stale:
            self._vaults.pop(k, None)
        if len(self._vaults) > self.max_sessions:
            for k, _ in sorted(self._vaults.items(), key=lambda kv: kv[1].last_used)[
                    : len(self._vaults) - self.max_sessions]:
                self._vaults.pop(k, None)

    def __len__(self) -> int:
        return len(self._vaults)


# -- Helpers -----------------------------------------------------------------

def term_detectors(terms: Optional[Iterable[str]], label: str = "NAME") -> List[Detector]:
    """Terms the operator declared identifying -- a customer's name, an employer,
    a project codename. Multi-word terms match whole AND word by word, so
    "Chandeep Samra" still masks a lone "Chandeep" later in the same sentence."""
    out: List[Detector] = []
    seen = set()

    def add(term: str) -> None:
        t = (term or "").strip()
        if len(t) < 3 or t.lower() in seen:
            return
        seen.add(t.lower())
        out.append(Detector(label, re.compile(
            r"(?<![\w@.'’-])" + re.escape(t) + r"(?![\w@'’-])", re.I)))

    for term in terms or ():
        whole = (term or "").strip()
        if not whole:
            continue
        add(whole)
        if re.search(r"\s", whole):
            for word in whole.split():
                if len(word) >= 4:
                    add(word)
    # Longest first: "Chandeep Samra" must win over "Chandeep".
    out.sort(key=lambda d: len(d.re.pattern), reverse=True)
    return out


def _is_token(s: str) -> bool:
    m = TOKEN_RE.fullmatch(s)
    return m is not None


def _token_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _apply(text: str, det: Detector, on_hit) -> str:
    spans = _token_spans(text)
    out: List[str] = []
    last = 0
    for m in det.re.finditer(text):
        captured = m.group(det.group) if det.group else m.group(0)
        if not captured:
            continue
        start = m.start(det.group) if det.group else m.start()
        if det.refine is not None:
            refined = det.refine(captured)
            if not refined:
                continue
            start += len(captured) - len(refined)
            captured = refined
        end = start + len(captured)
        # A span already covered by a placeholder from an earlier detector must
        # not be re-matched -- "[NATIONAL_ID_1]" contains digits.
        if any(s < end and e > start for s, e in spans) or _is_token(captured):
            continue
        if det.validate and not det.validate(captured):
            continue
        if start < last:
            continue
        out.append(text[last:start])
        out.append(on_hit(det.label, captured))
        last = end
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


def _active(entities: Optional[Sequence[str]], allow: Optional[Sequence[str]], label: str) -> bool:
    if allow and label in allow:
        return False
    if entities:
        return label in entities
    return True


# -- Redaction ---------------------------------------------------------------

def redact_text(text: str, vault: Vault, strict: bool = False,
                terms: Optional[Sequence[str]] = None,
                entities: Optional[Sequence[str]] = None,
                allow: Optional[Sequence[str]] = None) -> Tuple[str, int]:
    """Replace every detected PII span in ``text`` with a vault placeholder.
    Returns (redacted text, number of spans masked)."""
    if not isinstance(text, str) or not text or vault.mode == "off":
        return (text or ""), 0
    before = vault.total()
    out = text

    # Structural detectors run first: an email containing the user's name must
    # be masked as one email, not sliced into [NAME_1].[EMAIL_1].
    for det in DETECTORS:
        if det.strict and not strict:
            continue
        if not _active(entities, allow, det.label):
            continue
        out = _apply(out, det, vault.token_for)
    for det in term_detectors(terms):
        if _active(entities, allow, det.label):
            out = _apply(out, det, vault.token_for)
    if strict and _active(entities, allow, "NAME"):
        out = _apply(out, NAME_HEURISTIC, vault.token_for)
    return out, vault.total() - before


def scan_text(text: str, strict: bool = False, terms: Optional[Sequence[str]] = None,
              entities: Optional[Sequence[str]] = None,
              allow: Optional[Sequence[str]] = None) -> List[str]:
    """Detect without replacing -- used by the egress tripwire to prove a request
    body is clean. Returns labels only; values are never returned or logged."""
    if not isinstance(text, str) or not text:
        return []
    found: List[str] = []

    def sink(label: str, _value: str) -> str:
        found.append(label)
        return "\x00"

    probe = text
    for det in DETECTORS:
        if det.strict and not strict:
            continue
        if not _active(entities, allow, det.label):
            continue
        probe = _apply(probe, det, sink)
    for det in term_detectors(terms):
        if _active(entities, allow, det.label):
            probe = _apply(probe, det, sink)
    if strict and _active(entities, allow, "NAME"):
        _apply(probe, NAME_HEURISTIC, sink)
    seen = set()
    return [x for x in found if not (x in seen or seen.add(x))]


# -- Streaming ---------------------------------------------------------------

MAX_TOKEN_LEN = 48
_PARTIAL_RE = re.compile(r"^\[[A-Z0-9_]*$")


class StreamRestorer:
    """The model streams "[EMA" then "IL_1] -- see attached". Restoring chunk by
    chunk would emit the fragments raw, so hold back any tail that could still
    become a placeholder and release it once it resolves (or cannot)."""

    def __init__(self, vault: Optional[Vault]) -> None:
        self.vault = vault
        self._buf = ""

    def push(self, chunk: Optional[str]) -> str:
        self._buf += "" if chunk is None else str(chunk)
        hold = ""
        open_at = self._buf.rfind("[")
        if open_at != -1 and self._buf.find("]", open_at) == -1:
            tail = self._buf[open_at:]
            if len(tail) <= MAX_TOKEN_LEN and _PARTIAL_RE.match(tail):
                hold = tail
                self._buf = self._buf[:open_at]
        out = self.vault.restore(self._buf) if self.vault else self._buf
        self._buf = hold
        return out or ""

    def flush(self) -> str:
        out = self.vault.restore(self._buf) if self.vault else self._buf
        self._buf = ""
        return out or ""
