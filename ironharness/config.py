"""Runtime settings and policy loading.

Settings (where the database lives, which port to bind) come from the
environment, because they change per deployment. Policy (what is allowed) comes
from a YAML file, because it is reviewed, versioned and diffed like code.

The policy file can be reloaded without dropping in-flight requests: the manager
swaps a fully-parsed, fully-validated Config in one assignment, and a bad file
leaves the running policy untouched.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

from .policy import Config, PolicyError, parse_config

DEFAULT_POLICY_PATH = "policy.yaml"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 4600
    db_path: str = "ironharness.db"
    policy_path: str = DEFAULT_POLICY_PATH
    admin_token: str = ""
    # HMAC secret for `mode: hash` placeholders. Must be stable across restarts
    # if you want hashed tokens to join across days -- set it explicitly.
    hash_secret: str = ""
    vault_ttl_seconds: int = 3600
    request_timeout: float = 120.0
    trust_proxy_headers: bool = False
    cors_origins: str = ""
    dashboard: bool = True

    @staticmethod
    def from_env(env: Optional[Dict[str, str]] = None) -> "Settings":
        e = env if env is not None else os.environ
        s = Settings()
        s.host = e.get("IRONHARNESS_HOST", s.host)
        s.port = int(e.get("IRONHARNESS_PORT", s.port))
        s.db_path = e.get("IRONHARNESS_DB", s.db_path)
        s.policy_path = e.get("IRONHARNESS_POLICY", s.policy_path)
        s.admin_token = e.get("IRONHARNESS_ADMIN_TOKEN", "")
        s.hash_secret = e.get("IRONHARNESS_HASH_SECRET", "")
        s.vault_ttl_seconds = int(e.get("IRONHARNESS_VAULT_TTL", s.vault_ttl_seconds))
        s.request_timeout = float(e.get("IRONHARNESS_TIMEOUT", s.request_timeout))
        s.trust_proxy_headers = e.get("IRONHARNESS_TRUST_PROXY", "").lower() in ("1", "true", "yes")
        s.cors_origins = e.get("IRONHARNESS_CORS", "")
        s.dashboard = e.get("IRONHARNESS_DASHBOARD", "1").lower() not in ("0", "false", "no")
        return s

    def ensure_admin_token(self) -> str:
        """A gateway with an open admin API is a gateway with no security at all.
        If the operator did not set one, mint an ephemeral token and print it --
        never silently run without auth."""
        if not self.admin_token:
            self.admin_token = "ihadm_" + secrets.token_urlsafe(24)
            self._generated = True
        return self.admin_token

    @property
    def admin_token_generated(self) -> bool:
        return getattr(self, "_generated", False)


class PolicyManager:
    """Holds the live Config and reloads it from disk on demand."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._config: Optional[Config] = None
        self._loaded_at: float = 0.0
        self._mtime: float = 0.0
        self.pricing_overrides: Dict[str, Any] = {}

    @property
    def config(self) -> Config:
        if self._config is None:
            self.load()
        return self._config           # type: ignore[return-value]

    @property
    def loaded_at(self) -> float:
        return self._loaded_at

    def load(self) -> Config:
        doc = self._read()
        cfg = parse_config(doc, source=self.path)
        with self._lock:
            self._config = cfg
            self._loaded_at = time.time()
            self._mtime = self._file_mtime()
            self.pricing_overrides = doc.get("pricing", {}) or {}
        return cfg

    def reload_if_changed(self) -> bool:
        if self._file_mtime() != self._mtime:
            self.load()
            return True
        return False

    def _file_mtime(self) -> float:
        try:
            return os.path.getmtime(self.path)
        except OSError:
            return 0.0

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            raise PolicyError(
                "policy file not found: %s -- run `ironharness init` to write a "
                "starter policy" % self.path)
        with open(self.path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        if not isinstance(doc, dict):
            raise PolicyError("%s: top level must be a mapping" % self.path)
        return doc


STARTER_POLICY = """\
# IronHarness policy -- the security contract for everything behind this gateway.
# Reload without restarting:  POST /admin/policy/reload
version: 1

# "keys" (default) requires an IronHarness API key per caller, which is what
# binds a request to a policy, a budget and an audit trail.
# "open" lets anything through on the default policy -- local development only.
auth_mode: keys

upstreams:
  anthropic:
    provider: anthropic
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
  openai:
    provider: openai
    base_url: https://api.openai.com
    api_key_env: OPENAI_API_KEY
  gemini:
    provider: gemini
    base_url: https://generativelanguage.googleapis.com
    api_key_env: GEMINI_API_KEY

# Rates are $/1M tokens. Anthropic list prices ship built in; add your own for
# any other model you route, otherwise its spend is reported as unknown.
pricing: {}
#  "gpt-4o":           {input: 2.50, output: 10.00}
#  "gemini-2.5-flash": {input: 0.30, output: 2.50}

policies:
  default:
    description: House rules. Every other policy extends this one.

    pii:
      enabled: true
      mode: reversible        # reversible | mask | hash | off
      strict: false           # also mask bare names/handles (more false positives)
      entities: []            # empty = every detector; else e.g. [EMAIL, PHONE, CARD]
      allow: []               # detectors to exempt, e.g. [IP] for a netops assistant
      terms: []               # always mask these, e.g. ["Project Bluebird"]
      restore_in_response: true
      tripwire: true          # refuse to forward a body that still carries PII

    guardrails:
      input:
        - id: prompt_injection
          type: injection
          action: block
          threshold: 0.6
        - id: invisible_text
          type: unicode
          action: redact
        - id: oversized
          type: max_length
          action: block
          max_chars: 400000
      output:
        - id: credential_leak
          type: secrets
          action: redact

    limits:
      requests_per_minute: 120
      tokens_per_day: 2000000
      usd_per_day: 50
      max_concurrent: 16

    observability:
      log_payloads: redacted   # none | redacted | full
      sample_rate: 1.0
      retention_days: 30

  # A stricter tenant: masks names too, blocks a topic, spends less.
  strict:
    extends: default
    pii:
      strict: true
      mode: reversible
    guardrails:
      input:
        - id: prompt_injection
          type: injection
          action: block
          threshold: 0.45
        - id: invisible_text
          type: unicode
          action: redact
        - id: restricted_topics
          type: topics
          action: block
          topics:
            legal_advice: ["sue", "lawsuit", "legal advice"]
            medical_advice: ["diagnose me", "prescribe"]
      output:
        - id: credential_leak
          type: secrets
          action: block
        - id: pii_in_output
          type: pii_leak
          action: warn
    limits:
      requests_per_minute: 30
      usd_per_day: 5

default_policy: default
"""
