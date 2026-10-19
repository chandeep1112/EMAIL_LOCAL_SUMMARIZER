"""Durable state: API keys, the request log, and a tamper-evident audit chain.

SQLite in WAL mode -- one file, no server, safe for the single-process gateway
and good for millions of rows. Point IRONHARNESS_DB at a volume in production.

Two invariants this module exists to hold:

1. A key is never stored. Only its SHA-256 and a display prefix are, so a stolen
   database cannot be replayed against the gateway.
2. Audit rows are hash-chained. Editing or deleting any row breaks every hash
   after it, which `verify_audit()` reports with the exact sequence number.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS keys (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  key_hash    TEXT NOT NULL UNIQUE,
  prefix      TEXT NOT NULL,
  policy      TEXT NOT NULL DEFAULT 'default',
  upstream    TEXT,
  disabled    INTEGER NOT NULL DEFAULT 0,
  created_at  REAL NOT NULL,
  last_used_at REAL,
  metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS requests (
  id              TEXT PRIMARY KEY,
  ts              REAL NOT NULL,
  key_id          TEXT,
  key_name        TEXT,
  policy          TEXT,
  upstream        TEXT,
  provider        TEXT,
  model           TEXT,
  path            TEXT,
  method          TEXT,
  outcome         TEXT NOT NULL,      -- allowed | blocked | error | limited
  status          INTEGER,
  upstream_status INTEGER,
  latency_ms      REAL,
  upstream_ms     REAL,
  input_tokens    INTEGER NOT NULL DEFAULT 0,
  output_tokens   INTEGER NOT NULL DEFAULT 0,
  cost_usd        REAL,               -- NULL = rates unknown for this model
  stream          INTEGER NOT NULL DEFAULT 0,
  session_id      TEXT,
  client_ip       TEXT,
  block_rule      TEXT,
  block_reason    TEXT,
  pii_masked      INTEGER NOT NULL DEFAULT 0,
  pii_input       TEXT NOT NULL DEFAULT '{}',
  pii_output      TEXT NOT NULL DEFAULT '{}',
  findings        TEXT NOT NULL DEFAULT '[]',
  request_text    TEXT,
  response_text   TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts     ON requests (ts DESC);
CREATE INDEX IF NOT EXISTS idx_requests_key_ts ON requests (key_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_requests_out    ON requests (outcome, ts DESC);

CREATE TABLE IF NOT EXISTS audit (
  seq       INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        REAL NOT NULL,
  event     TEXT NOT NULL,
  actor     TEXT,
  subject   TEXT,
  payload   TEXT NOT NULL DEFAULT '{}',
  prev_hash TEXT NOT NULL,
  hash      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit (ts DESC);
"""

GENESIS_HASH = "0" * 64
KEY_PREFIX = "ih_"


@dataclass
class KeyRecord:
    id: str
    name: str
    prefix: str
    policy: str
    upstream: Optional[str]
    disabled: bool
    created_at: float
    last_used_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["created_at"] = self.created_at
        return d


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(24).replace("-", "").replace("_", "")[:32]


class Store:
    def __init__(self, path: str = "ironharness.db") -> None:
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._shared: Optional[sqlite3.Connection] = None
        if path == ":memory:":
            self._shared = self._connect()
        with self.conn() as c:
            c.executescript(_SCHEMA)
            c.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                      (str(SCHEMA_VERSION),))

    # -- connection ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, check_same_thread=False, timeout=15.0)
        c.row_factory = sqlite3.Row
        if self.path != ":memory:":
            c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._connect()
            self._local.conn = c
        return c

    def close(self) -> None:
        for c in (self._shared, getattr(self._local, "conn", None)):
            if c is not None:
                c.close()
        self._shared = None
        self._local = threading.local()

    # -- keys ---------------------------------------------------------------
    def create_key(self, name: str, policy: str = "default", upstream: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   actor: str = "admin") -> Tuple[KeyRecord, str]:
        plaintext = new_key()
        rec = KeyRecord(id=uuid.uuid4().hex[:12], name=name, prefix=plaintext[:11],
                        policy=policy, upstream=upstream, disabled=False,
                        created_at=time.time(), metadata=metadata or {})
        with self._write_lock:
            self.conn().execute(
                "INSERT INTO keys (id,name,key_hash,prefix,policy,upstream,disabled,"
                "created_at,metadata) VALUES (?,?,?,?,?,?,0,?,?)",
                (rec.id, rec.name, hash_key(plaintext), rec.prefix, rec.policy,
                 rec.upstream, rec.created_at, json.dumps(rec.metadata)))
            self.conn().commit()
        self.audit("key.created", actor=actor, subject=rec.id,
                   payload={"name": name, "policy": policy, "prefix": rec.prefix})
        return rec, plaintext

    def lookup_key(self, plaintext: str) -> Optional[KeyRecord]:
        row = self.conn().execute(
            "SELECT * FROM keys WHERE key_hash = ?", (hash_key(plaintext),)).fetchone()
        return self._key_from_row(row) if row else None

    def get_key(self, key_id: str) -> Optional[KeyRecord]:
        row = self.conn().execute("SELECT * FROM keys WHERE id = ?", (key_id,)).fetchone()
        return self._key_from_row(row) if row else None

    def list_keys(self) -> List[KeyRecord]:
        rows = self.conn().execute("SELECT * FROM keys ORDER BY created_at DESC").fetchall()
        return [self._key_from_row(r) for r in rows]

    def touch_key(self, key_id: str) -> None:
        with self._write_lock:
            self.conn().execute("UPDATE keys SET last_used_at = ? WHERE id = ?",
                                (time.time(), key_id))
            self.conn().commit()

    def set_key_disabled(self, key_id: str, disabled: bool, actor: str = "admin") -> bool:
        with self._write_lock:
            cur = self.conn().execute("UPDATE keys SET disabled = ? WHERE id = ?",
                                      (1 if disabled else 0, key_id))
            self.conn().commit()
        if cur.rowcount:
            self.audit("key.disabled" if disabled else "key.enabled",
                       actor=actor, subject=key_id)
        return bool(cur.rowcount)

    def delete_key(self, key_id: str, actor: str = "admin") -> bool:
        with self._write_lock:
            cur = self.conn().execute("DELETE FROM keys WHERE id = ?", (key_id,))
            self.conn().commit()
        if cur.rowcount:
            self.audit("key.deleted", actor=actor, subject=key_id)
        return bool(cur.rowcount)

    @staticmethod
    def _key_from_row(row: sqlite3.Row) -> KeyRecord:
        return KeyRecord(
            id=row["id"], name=row["name"], prefix=row["prefix"], policy=row["policy"],
            upstream=row["upstream"], disabled=bool(row["disabled"]),
            created_at=row["created_at"], last_used_at=row["last_used_at"],
            metadata=json.loads(row["metadata"] or "{}"))

    # -- request log --------------------------------------------------------
    def record_request(self, rec: Dict[str, Any]) -> None:
        cols = ("id", "ts", "key_id", "key_name", "policy", "upstream", "provider", "model",
                "path", "method", "outcome", "status", "upstream_status", "latency_ms",
                "upstream_ms", "input_tokens", "output_tokens", "cost_usd", "stream",
                "session_id", "client_ip", "block_rule", "block_reason", "pii_masked",
                "pii_input", "pii_output", "findings", "request_text", "response_text")
        values = []
        for c in cols:
            v = rec.get(c)
            if c in ("pii_input", "pii_output", "findings") and not isinstance(v, str):
                v = json.dumps(v or ({} if c != "findings" else []))
            if c == "stream":
                v = 1 if v else 0
            values.append(v)
        with self._write_lock:
            self.conn().execute(
                "INSERT OR REPLACE INTO requests (%s) VALUES (%s)"
                % (",".join(cols), ",".join("?" * len(cols))), values)
            self.conn().commit()

    def requests(self, limit: int = 100, offset: int = 0, key_id: Optional[str] = None,
                 outcome: Optional[str] = None, since: Optional[float] = None,
                 model: Optional[str] = None, q: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM requests WHERE 1=1"
        args: List[Any] = []
        if key_id:
            sql += " AND key_id = ?"; args.append(key_id)
        if outcome:
            sql += " AND outcome = ?"; args.append(outcome)
        if model:
            sql += " AND model = ?"; args.append(model)
        if since:
            sql += " AND ts >= ?"; args.append(since)
        if q:
            sql += " AND (model LIKE ? OR path LIKE ? OR block_rule LIKE ? OR key_name LIKE ?)"
            args += ["%%%s%%" % q] * 4
        sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        return [self._request_from_row(r) for r in self.conn().execute(sql, args).fetchall()]

    def request(self, request_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn().execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        return self._request_from_row(row) if row else None

    @staticmethod
    def _request_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for k in ("pii_input", "pii_output", "findings"):
            try:
                d[k] = json.loads(d.get(k) or ("[]" if k == "findings" else "{}"))
            except (TypeError, ValueError):
                d[k] = [] if k == "findings" else {}
        d["stream"] = bool(d.get("stream"))
        return d

    # -- usage / stats ------------------------------------------------------
    def usage(self, key_id: Optional[str], since: float) -> Dict[str, float]:
        sql = ("SELECT COUNT(*) n, COALESCE(SUM(input_tokens),0) it, "
               "COALESCE(SUM(output_tokens),0) ot, COALESCE(SUM(cost_usd),0) c "
               "FROM requests WHERE ts >= ? AND outcome != 'limited'")
        args: List[Any] = [since]
        if key_id:
            sql += " AND key_id = ?"; args.append(key_id)
        r = self.conn().execute(sql, args).fetchone()
        return {"requests": int(r["n"]), "input_tokens": int(r["it"]),
                "output_tokens": int(r["ot"]), "tokens": int(r["it"]) + int(r["ot"]),
                "cost_usd": float(r["c"])}

    def stats(self, since: float) -> Dict[str, Any]:
        c = self.conn()
        totals = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(input_tokens),0) it, COALESCE(SUM(output_tokens),0) ot,"
            " COALESCE(SUM(cost_usd),0) cost, COALESCE(SUM(pii_masked),0) pii,"
            " COALESCE(AVG(latency_ms),0) lat,"
            " SUM(CASE WHEN outcome='blocked' THEN 1 ELSE 0 END) blocked,"
            " SUM(CASE WHEN outcome='limited' THEN 1 ELSE 0 END) limited,"
            " SUM(CASE WHEN outcome='error' THEN 1 ELSE 0 END) errors,"
            " SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END) unpriced"
            " FROM requests WHERE ts >= ?", (since,)).fetchone()
        p95_row = c.execute(
            "SELECT latency_ms FROM requests WHERE ts >= ? AND latency_ms IS NOT NULL"
            " ORDER BY latency_ms LIMIT 1 OFFSET"
            " (SELECT CAST(COUNT(*)*0.95 AS INT) FROM requests WHERE ts >= ?"
            "  AND latency_ms IS NOT NULL)", (since, since)).fetchone()
        by_model = [dict(r) for r in c.execute(
            "SELECT model, COUNT(*) n, COALESCE(SUM(cost_usd),0) cost,"
            " COALESCE(SUM(input_tokens+output_tokens),0) tokens"
            " FROM requests WHERE ts >= ? GROUP BY model ORDER BY n DESC LIMIT 20",
            (since,)).fetchall()]
        by_key = [dict(r) for r in c.execute(
            "SELECT key_id, key_name, COUNT(*) n, COALESCE(SUM(cost_usd),0) cost"
            " FROM requests WHERE ts >= ? GROUP BY key_id ORDER BY n DESC LIMIT 20",
            (since,)).fetchall()]
        rules = [dict(r) for r in c.execute(
            "SELECT block_rule rule, COUNT(*) n FROM requests"
            " WHERE ts >= ? AND block_rule IS NOT NULL GROUP BY block_rule"
            " ORDER BY n DESC LIMIT 20", (since,)).fetchall()]
        series = [dict(r) for r in c.execute(
            "SELECT CAST(ts/3600 AS INT)*3600 bucket, COUNT(*) n,"
            " SUM(CASE WHEN outcome='blocked' THEN 1 ELSE 0 END) blocked,"
            " COALESCE(SUM(cost_usd),0) cost, COALESCE(SUM(pii_masked),0) pii"
            " FROM requests WHERE ts >= ? GROUP BY bucket ORDER BY bucket", (since,)).fetchall()]
        entities: Dict[str, int] = {}
        for row in c.execute("SELECT pii_input, pii_output FROM requests WHERE ts >= ?"
                             " AND pii_masked > 0 LIMIT 5000", (since,)).fetchall():
            for blob in (row["pii_input"], row["pii_output"]):
                try:
                    for k, v in (json.loads(blob or "{}") or {}).items():
                        entities[k] = entities.get(k, 0) + int(v)
                except (TypeError, ValueError):
                    continue
        return {
            "since": since,
            "requests": int(totals["n"]),
            "blocked": int(totals["blocked"] or 0),
            "limited": int(totals["limited"] or 0),
            "errors": int(totals["errors"] or 0),
            "input_tokens": int(totals["it"]),
            "output_tokens": int(totals["ot"]),
            "cost_usd": round(float(totals["cost"]), 6),
            "unpriced_requests": int(totals["unpriced"] or 0),
            "pii_masked": int(totals["pii"] or 0),
            "latency_ms_avg": round(float(totals["lat"] or 0), 1),
            "latency_ms_p95": round(float(p95_row["latency_ms"]), 1) if p95_row else 0.0,
            "by_model": by_model, "by_key": by_key, "top_rules": rules,
            "series": series,
            "entities": dict(sorted(entities.items(), key=lambda kv: -kv[1])),
        }

    def purge(self, older_than_ts: float) -> int:
        with self._write_lock:
            cur = self.conn().execute("DELETE FROM requests WHERE ts < ?", (older_than_ts,))
            self.conn().commit()
        return cur.rowcount

    def scrub_payloads(self, older_than_ts: float) -> int:
        """Drop stored prompt/response text while keeping the metrics rows."""
        with self._write_lock:
            cur = self.conn().execute(
                "UPDATE requests SET request_text = NULL, response_text = NULL"
                " WHERE ts < ? AND (request_text IS NOT NULL OR response_text IS NOT NULL)",
                (older_than_ts,))
            self.conn().commit()
        return cur.rowcount

    # -- audit chain --------------------------------------------------------
    def audit(self, event: str, actor: str = "system", subject: str = "",
              payload: Optional[Dict[str, Any]] = None) -> int:
        body = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))
        ts = time.time()
        with self._write_lock:
            row = self.conn().execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS_HASH
            digest = self._chain_hash(prev, ts, event, actor, subject, body)
            cur = self.conn().execute(
                "INSERT INTO audit (ts,event,actor,subject,payload,prev_hash,hash)"
                " VALUES (?,?,?,?,?,?,?)", (ts, event, actor, subject, body, prev, digest))
            self.conn().commit()
            return int(cur.lastrowid)

    @staticmethod
    def _chain_hash(prev: str, ts: float, event: str, actor: str, subject: str,
                    payload: str) -> str:
        blob = "|".join([prev, "%.6f" % ts, event, actor or "", subject or "", payload])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def audit_entries(self, limit: int = 200, offset: int = 0,
                      since: Optional[float] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM audit WHERE 1=1"
        args: List[Any] = []
        if since:
            sql += " AND ts >= ?"; args.append(since)
        sql += " ORDER BY seq DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        out = []
        for r in self.conn().execute(sql, args).fetchall():
            d = dict(r)
            try:
                d["payload"] = json.loads(d["payload"])
            except (TypeError, ValueError):
                pass
            out.append(d)
        return out

    def verify_audit(self) -> Dict[str, Any]:
        prev = GENESIS_HASH
        checked = 0
        for r in self.conn().execute("SELECT * FROM audit ORDER BY seq ASC"):
            expect = self._chain_hash(prev, r["ts"], r["event"], r["actor"] or "",
                                      r["subject"] or "", r["payload"])
            if r["prev_hash"] != prev or r["hash"] != expect:
                return {"ok": False, "checked": checked, "broken_at": int(r["seq"]),
                        "reason": "hash mismatch -- row %d was altered, or a row before it"
                                  " was deleted" % int(r["seq"])}
            prev = r["hash"]
            checked += 1
        return {"ok": True, "checked": checked, "head": prev}

    # -- export -------------------------------------------------------------
    def export_rows(self, since: Optional[float] = None,
                    table: str = "requests") -> Iterable[Dict[str, Any]]:
        if table not in ("requests", "audit"):
            raise ValueError("table must be 'requests' or 'audit'")
        sql = "SELECT * FROM %s" % table
        args: List[Any] = []
        if since:
            sql += " WHERE ts >= ?"; args.append(since)
        sql += " ORDER BY ts ASC"
        for row in self.conn().execute(sql, args):
            yield self._request_from_row(row) if table == "requests" else dict(row)
