"""Anthropic Messages API wire format (/v1/messages)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import Provider, Ref, collect_all_strings, collect_strings


class AnthropicProvider(Provider):
    name = "anthropic"
    paths = ("/v1/messages", "/v1/messages/count_tokens", "/v1/models",
             "/v1/messages/batches", "/v1/complete")

    def auth_headers(self, api_key: str) -> Dict[str, str]:
        # Anthropic reads x-api-key, not Authorization. An OAuth token still
        # goes on Authorization, so a caller-supplied bearer is preserved by the
        # gateway when no upstream key is configured.
        return {"x-api-key": api_key} if api_key else {}

    def request_refs(self, body: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        system = body.get("system")
        if isinstance(system, str):
            refs.append((body, "system"))
        elif isinstance(system, list):
            collect_strings(system, refs, ("text",))
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                refs.append((msg, "content"))
            elif isinstance(content, list):
                # text blocks, tool_result content, and the string values inside
                # a replayed tool_use input all carry caller text.
                collect_strings(content, refs, ("text", "content"))
        if isinstance(body.get("prompt"), str):            # legacy /v1/complete
            refs.append((body, "prompt"))
        return refs

    def response_refs(self, body: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        for block in body.get("content") or []:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                refs.append((block, "text"))
            if isinstance(block.get("thinking"), str):
                refs.append((block, "thinking"))
            if isinstance(block.get("input"), (dict, list)):
                # A tool_use input is the caller's own schema: any string in it
                # may carry a placeholder that has to be restored.
                collect_all_strings(block["input"], refs)
        if isinstance(body.get("completion"), str):
            refs.append((body, "completion"))
        return refs

    def usage(self, body: Dict[str, Any]) -> Tuple[int, int]:
        u = body.get("usage") or {}
        in_tok = (int(u.get("input_tokens") or 0)
                  + int(u.get("cache_creation_input_tokens") or 0)
                  + int(u.get("cache_read_input_tokens") or 0))
        return in_tok, int(u.get("output_tokens") or 0)

    def stream_refs(self, event: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        etype = event.get("type")
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            for key in ("text", "partial_json", "thinking"):
                if isinstance(delta.get(key), str):
                    refs.append((delta, key))
        elif etype == "content_block_start":
            block = event.get("content_block") or {}
            for key in ("text", "thinking"):
                if isinstance(block.get(key), str):
                    refs.append((block, key))
        return refs

    def stream_usage(self, event: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        if event.get("type") == "message_start":
            msg = event.get("message") or {}
            if msg.get("usage"):
                return self.usage(msg)
        if event.get("type") == "message_delta" and event.get("usage"):
            u = event["usage"]
            return int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        return None

    def stream_done(self, raw: str) -> bool:
        return False   # Anthropic signals completion with a message_stop event

    def error_body(self, message: str, code: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "error",
                "error": {"type": "ironharness_blocked", "message": message,
                          "code": code, "ironharness": detail}}
