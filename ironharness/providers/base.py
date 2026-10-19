"""Provider adapters.

IronHarness is a transparent proxy: it forwards the caller's body to the caller's
model, byte for byte, except for the text spans it is asked to police. An adapter's
only job is to say *where the text lives* in a given wire format -- as mutable
references, so the gateway can read a span, rewrite it, and put it back without
reconstructing (and therefore subtly corrupting) the request.

Adding a provider means implementing this class. It needs no knowledge of PII,
policy, limits or storage.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

# A reference to a mutable string inside a parsed JSON body: (container, key).
Ref = Tuple[Any, Any]


def read(ref: Ref) -> str:
    container, key = ref
    value = container[key]
    return value if isinstance(value, str) else ""


def write(ref: Ref, value: str) -> None:
    container, key = ref
    container[key] = value


def collect_strings(node: Any, refs: List[Ref], keys: Iterable[str]) -> None:
    """Depth-first sweep for the named string keys -- used for nested shapes
    (tool results, content part lists) whose depth varies between API versions."""
    wanted = tuple(keys)
    if isinstance(node, dict):
        for k, v in node.items():
            if k in wanted and isinstance(v, str):
                refs.append((node, k))
            else:
                collect_strings(v, refs, wanted)
    elif isinstance(node, list):
        for item in node:
            collect_strings(item, refs, wanted)


def collect_all_strings(node: Any, refs: List[Ref]) -> None:
    """Every string value anywhere under `node` -- used for free-form objects
    such as a tool call's arguments, where key names are the caller's to choose."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                refs.append((node, k))
            else:
                collect_all_strings(v, refs)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                refs.append((node, i))
            else:
                collect_all_strings(v, refs)


class Provider:
    name = "base"
    # Path prefixes this adapter claims when no upstream is named explicitly.
    paths: Tuple[str, ...] = ()

    # -- request ------------------------------------------------------------
    def request_refs(self, body: Dict[str, Any]) -> List[Ref]:
        raise NotImplementedError

    def model(self, body: Dict[str, Any], path: str = "") -> str:
        return str(body.get("model") or "")

    def is_stream(self, body: Dict[str, Any], path: str = "") -> bool:
        return bool(body.get("stream"))

    def auth_headers(self, api_key: str) -> Dict[str, str]:
        return {"authorization": "Bearer " + api_key} if api_key else {}

    # -- response -----------------------------------------------------------
    def response_refs(self, body: Dict[str, Any]) -> List[Ref]:
        raise NotImplementedError

    def usage(self, body: Dict[str, Any]) -> Tuple[int, int]:
        """(input_tokens, output_tokens); zeros when the provider did not say."""
        return 0, 0

    # -- streaming ----------------------------------------------------------
    def stream_refs(self, event: Dict[str, Any]) -> List[Ref]:
        """Mutable text references inside one decoded SSE event."""
        return []

    def stream_usage(self, event: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        return None

    def stream_done(self, raw: str) -> bool:
        return raw.strip() == "[DONE]"

    # -- errors -------------------------------------------------------------
    def error_body(self, message: str, code: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        return {"error": {"message": message, "type": "ironharness_blocked",
                          "code": code, "ironharness": detail}}
