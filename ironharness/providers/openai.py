"""OpenAI-compatible wire format (also: Azure OpenAI, Groq, Together, vLLM,
Ollama, OpenRouter -- anything that speaks /v1/chat/completions)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import Provider, Ref, collect_strings


class OpenAIProvider(Provider):
    name = "openai"
    paths = ("/v1/chat/completions", "/v1/completions", "/v1/responses",
             "/v1/embeddings", "/v1/models", "/v1/moderations")

    def request_refs(self, body: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        for msg in body.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                refs.append((msg, "content"))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        refs.append((part, "text"))
            # assistant turns replayed from a previous call carry tool arguments
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                    refs.append((fn, "arguments"))
        if isinstance(body.get("prompt"), str):
            refs.append((body, "prompt"))
        elif isinstance(body.get("prompt"), list):
            for i, p in enumerate(body["prompt"]):
                if isinstance(p, str):
                    refs.append((body["prompt"], i))
        if isinstance(body.get("input"), str):           # /v1/responses, embeddings
            refs.append((body, "input"))
        elif isinstance(body.get("input"), list):
            collect_strings(body["input"], refs, ("text", "input_text", "content"))
        if isinstance(body.get("instructions"), str):     # /v1/responses system slot
            refs.append((body, "instructions"))
        return refs

    def response_refs(self, body: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        for choice in body.get("choices") or []:
            msg = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(msg, dict):
                if isinstance(msg.get("content"), str):
                    refs.append((msg, "content"))
                if isinstance(msg.get("reasoning_content"), str):
                    refs.append((msg, "reasoning_content"))
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function") if isinstance(call, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        refs.append((fn, "arguments"))
            if isinstance(choice, dict) and isinstance(choice.get("text"), str):
                refs.append((choice, "text"))
        for item in body.get("output") or []:             # /v1/responses
            collect_strings(item, refs, ("text",))
        return refs

    def usage(self, body: Dict[str, Any]) -> Tuple[int, int]:
        u = body.get("usage") or {}
        return (int(u.get("prompt_tokens") or u.get("input_tokens") or 0),
                int(u.get("completion_tokens") or u.get("output_tokens") or 0))

    def stream_refs(self, event: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        for choice in event.get("choices") or []:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                for key in ("content", "reasoning_content"):
                    if isinstance(delta.get(key), str):
                        refs.append((delta, key))
                for call in delta.get("tool_calls") or []:
                    fn = call.get("function") if isinstance(call, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        refs.append((fn, "arguments"))
            if isinstance(choice, dict) and isinstance(choice.get("text"), str):
                refs.append((choice, "text"))
        if isinstance(event.get("delta"), str):           # /v1/responses stream
            refs.append((event, "delta"))
        return refs

    def stream_usage(self, event: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        if event.get("usage"):
            return self.usage(event)
        return None
