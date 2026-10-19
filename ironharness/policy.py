"""Policy: the declarative contract an operator writes once and the gateway
enforces on every request.

A policy is plain data. It is loaded from YAML, validated on load (so a typo
fails at boot, not at 3am on a customer request), and attached to an API key.
Nothing in this module reaches out to the network or the database -- it is the
part of IronHarness you can unit-test in a millisecond.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pii.patterns import ALL_LABELS
from .pii.redact import MODES

ACTIONS = ("block", "redact", "warn", "allow")
DIRECTIONS = ("input", "output")
RULE_TYPES = ("injection", "topics", "regex", "secrets", "pii_leak", "unicode",
              "max_length", "model_allowlist")


class PolicyError(ValueError):
    """Raised for an invalid policy file. Always names the offending path."""


def _req(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise PolicyError("%s: missing required field %r" % (where, key))
    return d[key]


def _one_of(value: Any, allowed: Sequence[str], where: str) -> str:
    if value not in allowed:
        raise PolicyError("%s: %r is not one of %s" % (where, value, ", ".join(allowed)))
    return value


@dataclass
class PIIPolicy:
    enabled: bool = True
    mode: str = "reversible"          # reversible | mask | hash | off
    strict: bool = False              # also run the aggressive detectors
    entities: List[str] = field(default_factory=list)   # empty = every detector
    allow: List[str] = field(default_factory=list)      # never mask these labels
    terms: List[str] = field(default_factory=list)      # always-mask custom terms
    restore_in_response: bool = True
    # The egress tripwire re-scans the outbound body after redaction. On a hit
    # the request is refused rather than forwarded: a redactor that silently
    # stopped working is worse than an outage.
    tripwire: bool = True
    session_header: str = "x-ironharness-session"

    @staticmethod
    def parse(d: Dict[str, Any], where: str) -> "PIIPolicy":
        p = PIIPolicy()
        if not isinstance(d, dict):
            raise PolicyError("%s: pii must be a mapping" % where)
        p.enabled = bool(d.get("enabled", True))
        p.mode = _one_of(d.get("mode", "reversible"), MODES, where + ".mode")
        p.strict = bool(d.get("strict", False))
        for key in ("entities", "allow"):
            vals = [str(x).upper() for x in d.get(key, []) or []]
            for v in vals:
                if v not in ALL_LABELS:
                    raise PolicyError("%s.%s: unknown entity %r (known: %s)"
                                      % (where, key, v, ", ".join(ALL_LABELS)))
            setattr(p, key, vals)
        p.terms = [str(x) for x in d.get("terms", []) or []]
        p.restore_in_response = bool(d.get("restore_in_response", True))
        p.tripwire = bool(d.get("tripwire", True))
        p.session_header = str(d.get("session_header", p.session_header)).lower()
        return p


@dataclass
class Rule:
    id: str
    type: str
    action: str = "block"
    threshold: float = 0.5
    # type-specific
    patterns: List[Any] = field(default_factory=list)      # regex
    topics: Dict[str, List[str]] = field(default_factory=dict)
    max_chars: int = 0
    models: List[str] = field(default_factory=list)
    message: str = ""

    @staticmethod
    def parse(d: Dict[str, Any], where: str) -> "Rule":
        if not isinstance(d, dict):
            raise PolicyError("%s: rule must be a mapping" % where)
        rtype = _one_of(_req(d, "type", where), RULE_TYPES, where + ".type")
        rule = Rule(
            id=str(d.get("id") or rtype),
            type=rtype,
            action=_one_of(d.get("action", "block"), ACTIONS, where + ".action"),
            threshold=float(d.get("threshold", 0.5)),
            max_chars=int(d.get("max_chars", 0) or 0),
            models=[str(m) for m in d.get("models", []) or []],
            message=str(d.get("message", "")),
        )
        raw = d.get("patterns", []) or []
        for pat in raw:
            try:
                rule.patterns.append(re.compile(str(pat), re.I))
            except re.error as exc:
                raise PolicyError("%s.patterns: bad regex %r (%s)" % (where, pat, exc))
        topics = d.get("topics", {}) or {}
        if topics and not isinstance(topics, dict):
            raise PolicyError("%s.topics: must be a mapping of name -> keywords" % where)
        for name, words in topics.items():
            rule.topics[str(name)] = [str(w).lower() for w in words or []]
        if rtype == "regex" and not rule.patterns:
            raise PolicyError("%s: a regex rule needs at least one pattern" % where)
        if rtype == "topics" and not rule.topics:
            raise PolicyError("%s: a topics rule needs at least one topic" % where)
        if rtype == "max_length" and rule.max_chars <= 0:
            raise PolicyError("%s: max_length needs a positive max_chars" % where)
        if rtype == "model_allowlist" and not rule.models:
            raise PolicyError("%s: model_allowlist needs a models list" % where)
        return rule


@dataclass
class Limits:
    requests_per_minute: int = 0      # 0 = unlimited
    requests_per_day: int = 0
    tokens_per_day: int = 0
    usd_per_day: float = 0.0
    usd_per_month: float = 0.0
    max_concurrent: int = 0

    @staticmethod
    def parse(d: Dict[str, Any], where: str) -> "Limits":
        if not isinstance(d, dict):
            raise PolicyError("%s: limits must be a mapping" % where)
        lim = Limits()
        for f in ("requests_per_minute", "requests_per_day", "tokens_per_day", "max_concurrent"):
            lim.__dict__[f] = int(d.get(f, 0) or 0)
        for f in ("usd_per_day", "usd_per_month"):
            lim.__dict__[f] = float(d.get(f, 0) or 0)
        return lim


@dataclass
class Observability:
    # "none" keeps only counts; "redacted" stores the masked text (what the model
    # saw); "full" stores the raw prompt and is refused unless explicitly
    # acknowledged, because it defeats the point of the product.
    log_payloads: str = "redacted"
    sample_rate: float = 1.0
    retention_days: int = 30
    i_understand_full_logging_stores_pii: bool = False

    @staticmethod
    def parse(d: Dict[str, Any], where: str) -> "Observability":
        if not isinstance(d, dict):
            raise PolicyError("%s: observability must be a mapping" % where)
        o = Observability()
        o.log_payloads = _one_of(d.get("log_payloads", "redacted"),
                                 ("none", "redacted", "full"), where + ".log_payloads")
        o.sample_rate = float(d.get("sample_rate", 1.0))
        o.retention_days = int(d.get("retention_days", 30) or 0)
        o.i_understand_full_logging_stores_pii = bool(
            d.get("i_understand_full_logging_stores_pii", False))
        if o.log_payloads == "full" and not o.i_understand_full_logging_stores_pii:
            raise PolicyError(
                "%s.log_payloads: 'full' stores unredacted prompts. Set "
                "i_understand_full_logging_stores_pii: true to confirm." % where)
        if not 0.0 <= o.sample_rate <= 1.0:
            raise PolicyError("%s.sample_rate: must be between 0 and 1" % where)
        return o


@dataclass
class Policy:
    name: str = "default"
    pii: PIIPolicy = field(default_factory=PIIPolicy)
    input_rules: List[Rule] = field(default_factory=list)
    output_rules: List[Rule] = field(default_factory=list)
    limits: Limits = field(default_factory=Limits)
    observability: Observability = field(default_factory=Observability)
    upstreams: List[str] = field(default_factory=list)   # empty = all
    fail_open: bool = False   # if a guardrail itself errors, allow or refuse?

    @staticmethod
    def parse(name: str, d: Dict[str, Any], where: str) -> "Policy":
        if not isinstance(d, dict):
            raise PolicyError("%s: policy must be a mapping" % where)
        unknown = set(d) - {"pii", "guardrails", "limits", "observability",
                            "upstreams", "fail_open", "extends", "description"}
        if unknown:
            raise PolicyError("%s: unknown key(s): %s" % (where, ", ".join(sorted(unknown))))
        g = d.get("guardrails", {}) or {}
        if not isinstance(g, dict):
            raise PolicyError("%s.guardrails: must be a mapping with input/output lists" % where)
        for k in g:
            if k not in DIRECTIONS:
                raise PolicyError("%s.guardrails: unknown direction %r" % (where, k))
        return Policy(
            name=name,
            pii=PIIPolicy.parse(d.get("pii", {}) or {}, where + ".pii"),
            input_rules=[Rule.parse(r, "%s.guardrails.input[%d]" % (where, i))
                         for i, r in enumerate(g.get("input", []) or [])],
            output_rules=[Rule.parse(r, "%s.guardrails.output[%d]" % (where, i))
                          for i, r in enumerate(g.get("output", []) or [])],
            limits=Limits.parse(d.get("limits", {}) or {}, where + ".limits"),
            observability=Observability.parse(d.get("observability", {}) or {},
                                              where + ".observability"),
            upstreams=[str(u) for u in d.get("upstreams", []) or []],
            fail_open=bool(d.get("fail_open", False)),
        )


@dataclass
class Upstream:
    name: str
    provider: str            # which wire format: openai | anthropic | gemini
    base_url: str
    api_key: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 120.0
    headers: Dict[str, str] = field(default_factory=dict)

    def resolve_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


@dataclass
class Config:
    upstreams: Dict[str, Upstream] = field(default_factory=dict)
    policies: Dict[str, Policy] = field(default_factory=dict)
    default_policy: str = "default"
    # open  = any caller may pass through, using the default policy
    # keys  = an IronHarness key is required (issue with `ironharness keys add`)
    auth_mode: str = "keys"
    source_path: Optional[str] = None

    def policy(self, name: Optional[str]) -> Policy:
        return (self.policies.get(name or "")
                or self.policies.get(self.default_policy)
                or Policy())

    def upstream_for(self, name: str) -> Optional[Upstream]:
        return self.upstreams.get(name)


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_config(doc: Dict[str, Any], source: Optional[str] = None) -> Config:
    if not isinstance(doc, dict):
        raise PolicyError("policy file must be a YAML mapping")
    version = doc.get("version", 1)
    if int(version) != 1:
        raise PolicyError("unsupported policy version %r (this build speaks 1)" % version)

    cfg = Config(source_path=source)
    cfg.auth_mode = _one_of(doc.get("auth_mode", "keys"), ("keys", "open"), "auth_mode")

    for name, u in (doc.get("upstreams", {}) or {}).items():
        where = "upstreams.%s" % name
        if not isinstance(u, dict):
            raise PolicyError("%s: must be a mapping" % where)
        provider = str(u.get("provider", name))
        if provider not in ("openai", "anthropic", "gemini"):
            raise PolicyError("%s.provider: %r is not a known wire format "
                              "(openai, anthropic, gemini)" % (where, provider))
        base_url = str(_req(u, "base_url", where)).rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise PolicyError("%s.base_url: must be an http(s) URL" % where)
        cfg.upstreams[name] = Upstream(
            name=name, provider=provider, base_url=base_url,
            api_key=str(u.get("api_key", "") or ""),
            api_key_env=str(u.get("api_key_env", "") or ""),
            timeout_seconds=float(u.get("timeout_seconds", 120)),
            headers={str(k): str(v) for k, v in (u.get("headers", {}) or {}).items()},
        )
    if not cfg.upstreams:
        raise PolicyError("upstreams: at least one upstream must be configured")

    raw_policies = doc.get("policies", {}) or {}
    if "default" not in raw_policies:
        raw_policies = dict(raw_policies)
        raw_policies["default"] = {}
    # `extends` lets a tenant policy inherit the house rules and override a field.
    resolved: Dict[str, Dict[str, Any]] = {}

    def resolve(name: str, seen: Tuple[str, ...] = ()) -> Dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name in seen:
            raise PolicyError("policies.%s: circular 'extends' chain" % name)
        body = raw_policies.get(name)
        if body is None:
            raise PolicyError("policies.%s: extended policy does not exist" % name)
        parent = body.get("extends")
        merged = _merge(resolve(str(parent), seen + (name,)), body) if parent else dict(body)
        merged.pop("extends", None)
        resolved[name] = merged
        return merged

    for name in raw_policies:
        cfg.policies[name] = Policy.parse(name, resolve(name), "policies." + name)

    cfg.default_policy = str(doc.get("default_policy", "default"))
    if cfg.default_policy not in cfg.policies:
        raise PolicyError("default_policy: %r is not defined" % cfg.default_policy)
    for pname, pol in cfg.policies.items():
        for up in pol.upstreams:
            if up not in cfg.upstreams:
                raise PolicyError("policies.%s.upstreams: unknown upstream %r" % (pname, up))
    return cfg
