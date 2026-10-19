"""Token pricing.

Only rates IronHarness can state with confidence ship in the table below --
Anthropic's public first-party list prices. Every other model costs "unknown"
until an operator fills it in, and an unknown cost is reported as unknown rather
than guessed: a budget built on invented numbers is worse than no budget.

Override or extend with a `pricing:` block in the policy file, or a pricing.yaml
next to it:

    pricing:
      "gpt-4o":            {input: 2.50, output: 10.00}
      "gemini-2.5-flash":  {input: 0.30, output: 2.50}

Rates are US dollars per million tokens.
"""

from __future__ import annotations

import fnmatch
from typing import Dict, Optional, Tuple

# Anthropic first-party API list prices, $/1M tokens. Verified 2026-06-24.
DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5-1": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class Pricing:
    def __init__(self, overrides: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.table: Dict[str, Tuple[float, float]] = dict(DEFAULT_PRICING)
        self.update(overrides or {})

    def update(self, overrides: Dict[str, Dict[str, float]]) -> None:
        for model, rates in (overrides or {}).items():
            if isinstance(rates, dict):
                self.table[str(model)] = (float(rates.get("input", 0) or 0),
                                          float(rates.get("output", 0) or 0))
            elif isinstance(rates, (list, tuple)) and len(rates) == 2:
                self.table[str(model)] = (float(rates[0]), float(rates[1]))

    def rates(self, model: str) -> Optional[Tuple[float, float]]:
        if not model:
            return None
        if model in self.table:
            return self.table[model]
        # Snapshot ids ("claude-haiku-4-5-20251001") and vendor prefixes
        # ("anthropic.claude-opus-5", "models/gemini-2.5-flash").
        bare = model.split("/")[-1]
        for prefix in ("anthropic.", "us.anthropic.", "eu.anthropic."):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
        if bare in self.table:
            return self.table[bare]
        for known in sorted(self.table, key=len, reverse=True):
            if bare.startswith(known) or fnmatch.fnmatch(bare, known):
                return self.table[known]
        return None

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Dollars for one call, or None when the model's rates are unknown."""
        r = self.rates(model)
        if r is None:
            return None
        return round((input_tokens / 1e6) * r[0] + (output_tokens / 1e6) * r[1], 8)

    def known(self, model: str) -> bool:
        return self.rates(model) is not None
