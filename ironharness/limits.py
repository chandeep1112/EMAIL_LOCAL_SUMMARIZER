"""Rate limits, spend budgets, and concurrency caps.

Rates are enforced in memory (a sliding window per key, microseconds per check).
Budgets are enforced against the request log, which is the only place that knows
what a key has actually spent -- but a SQL query per request would be silly, so
each key's daily and monthly spend is cached for a few seconds and topped up
in memory as requests complete.

The tradeoff, stated plainly: a burst of concurrent requests can overshoot a
budget by whatever lands inside one cache window. Budgets are cost controls, not
payment authorisation -- for a hard ceiling, set `max_concurrent` as well.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from .policy import Limits
from .store import Store

DAY = 86400.0


@dataclass
class LimitVerdict:
    allowed: bool
    reason: str = ""
    limit: str = ""
    retry_after: int = 0
    detail: Optional[Dict[str, float]] = None


def _month_start(now: float) -> float:
    t = time.gmtime(now)
    return time.mktime((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 1, 0))


class Limiter:
    def __init__(self, store: Store, cache_ttl: float = 5.0) -> None:
        self.store = store
        self.cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._minute: Dict[str, Deque[float]] = {}
        self._day_reqs: Dict[str, Deque[float]] = {}
        self._concurrent: Dict[str, int] = {}
        # key -> (fetched_at, day_usage, month_usage) plus in-memory deltas
        self._usage: Dict[str, Tuple[float, Dict[str, float], Dict[str, float]]] = {}
        self._delta: Dict[str, Dict[str, float]] = {}

    # -- rates --------------------------------------------------------------
    def _trim(self, dq: Deque[float], window: float, now: float) -> None:
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()

    def check(self, key_id: str, limits: Limits, now: Optional[float] = None) -> LimitVerdict:
        now = now or time.time()
        key_id = key_id or "anonymous"
        with self._lock:
            if limits.requests_per_minute:
                dq = self._minute.setdefault(key_id, deque())
                self._trim(dq, 60.0, now)
                if len(dq) >= limits.requests_per_minute:
                    wait = max(1, int(60 - (now - dq[0])) + 1)
                    return LimitVerdict(False, "Rate limit exceeded: %d requests/minute."
                                        % limits.requests_per_minute,
                                        "requests_per_minute", wait,
                                        {"used": len(dq), "limit": limits.requests_per_minute})
            if limits.requests_per_day:
                dq = self._day_reqs.setdefault(key_id, deque())
                self._trim(dq, DAY, now)
                if len(dq) >= limits.requests_per_day:
                    return LimitVerdict(False, "Daily request limit reached (%d)."
                                        % limits.requests_per_day, "requests_per_day",
                                        int(DAY - (now - dq[0])) + 1,
                                        {"used": len(dq), "limit": limits.requests_per_day})
            if limits.max_concurrent:
                if self._concurrent.get(key_id, 0) >= limits.max_concurrent:
                    return LimitVerdict(False, "Too many concurrent requests (%d)."
                                        % limits.max_concurrent, "max_concurrent", 1,
                                        {"used": self._concurrent.get(key_id, 0),
                                         "limit": limits.max_concurrent})

        if limits.tokens_per_day or limits.usd_per_day or limits.usd_per_month:
            day, month = self._usage_for(key_id, now)
            if limits.tokens_per_day and day["tokens"] >= limits.tokens_per_day:
                return LimitVerdict(False, "Daily token budget exhausted (%d)."
                                    % limits.tokens_per_day, "tokens_per_day",
                                    self._seconds_to_midnight(now),
                                    {"used": day["tokens"], "limit": limits.tokens_per_day})
            if limits.usd_per_day and day["cost_usd"] >= limits.usd_per_day:
                return LimitVerdict(False, "Daily spend budget exhausted ($%.2f)."
                                    % limits.usd_per_day, "usd_per_day",
                                    self._seconds_to_midnight(now),
                                    {"used": round(day["cost_usd"], 4),
                                     "limit": limits.usd_per_day})
            if limits.usd_per_month and month["cost_usd"] >= limits.usd_per_month:
                return LimitVerdict(False, "Monthly spend budget exhausted ($%.2f)."
                                    % limits.usd_per_month, "usd_per_month", 3600,
                                    {"used": round(month["cost_usd"], 4),
                                     "limit": limits.usd_per_month})
        return LimitVerdict(True)

    @staticmethod
    def _seconds_to_midnight(now: float) -> int:
        return int(DAY - (now % DAY)) + 1

    def note_request(self, key_id: str, now: Optional[float] = None) -> None:
        """Called once a request is admitted -- it counts against the rate window
        whether or not the upstream ends up answering."""
        now = now or time.time()
        key_id = key_id or "anonymous"
        with self._lock:
            self._minute.setdefault(key_id, deque()).append(now)
            self._day_reqs.setdefault(key_id, deque()).append(now)

    # -- concurrency --------------------------------------------------------
    def enter(self, key_id: str) -> None:
        with self._lock:
            self._concurrent[key_id] = self._concurrent.get(key_id, 0) + 1

    def leave(self, key_id: str) -> None:
        with self._lock:
            self._concurrent[key_id] = max(0, self._concurrent.get(key_id, 0) - 1)

    # -- budgets ------------------------------------------------------------
    def _usage_for(self, key_id: str, now: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        with self._lock:
            cached = self._usage.get(key_id)
            fresh = cached and (now - cached[0]) < self.cache_ttl
            if not fresh:
                day = self.store.usage(key_id, now - DAY)
                month = self.store.usage(key_id, _month_start(now))
                self._usage[key_id] = (now, day, month)
                self._delta.pop(key_id, None)
            else:
                day, month = cached[1], cached[2]
            delta = self._delta.get(key_id)
            if delta:
                day = dict(day); month = dict(month)
                for field in ("tokens", "cost_usd", "requests"):
                    day[field] = day.get(field, 0) + delta.get(field, 0)
                    month[field] = month.get(field, 0) + delta.get(field, 0)
            return day, month

    def record_usage(self, key_id: str, tokens: int = 0, cost_usd: float = 0.0) -> None:
        with self._lock:
            d = self._delta.setdefault(key_id, {"tokens": 0.0, "cost_usd": 0.0, "requests": 0.0})
            d["tokens"] += tokens
            d["cost_usd"] += cost_usd or 0.0
            d["requests"] += 1

    def snapshot(self, key_id: str, limits: Limits) -> Dict[str, object]:
        now = time.time()
        day, month = self._usage_for(key_id, now)
        with self._lock:
            minute = len(self._minute.get(key_id, ()))
        return {
            "requests_last_minute": minute,
            "requests_per_minute_limit": limits.requests_per_minute or None,
            "tokens_today": int(day.get("tokens", 0)),
            "tokens_per_day_limit": limits.tokens_per_day or None,
            "usd_today": round(day.get("cost_usd", 0.0), 4),
            "usd_per_day_limit": limits.usd_per_day or None,
            "usd_this_month": round(month.get("cost_usd", 0.0), 4),
            "usd_per_month_limit": limits.usd_per_month or None,
            "concurrent": self._concurrent.get(key_id, 0),
        }
