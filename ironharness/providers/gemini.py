"""Google Gemini generateContent wire format."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import Provider, Ref, collect_strings


class GeminiProvider(Provider):
    name = "gemini"
    paths = ("/v1beta/", "/v1/models/")

    def auth_headers(self, api_key: str) -> Dict[str, str]:
        return {"x-goog-api-key": api_key} if api_key else {}

    def model(self, body: Dict[str, Any], path: str = "") -> str:
        if body.get("model"):
            return str(body["model"])
        # /v1beta/models/gemini-2.5-flash:generateContent
        if "/models/" in path:
            tail = path.split("/models/", 1)[1]
            return tail.split(":", 1)[0]
        return ""

    def is_stream(self, body: Dict[str, Any], path: str = "") -> bool:
        return ":streamGenerateContent" in path or bool(body.get("stream"))

    def request_refs(self, body: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        for key in ("contents", "systemInstruction", "system_instruction"):
            if key in body:
                collect_strings(body[key], refs, ("text",))
        return refs

    def response_refs(self, body: Dict[str, Any]) -> List[Ref]:
        refs: List[Ref] = []
        for cand in body.get("candidates") or []:
            collect_strings(cand.get("content"), refs, ("text",))
        return refs

    def usage(self, body: Dict[str, Any]) -> Tuple[int, int]:
        u = body.get("usageMetadata") or {}
        return int(u.get("promptTokenCount") or 0), int(u.get("candidatesTokenCount") or 0)

    def stream_refs(self, event: Dict[str, Any]) -> List[Ref]:
        return self.response_refs(event)

    def stream_usage(self, event: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        return self.usage(event) if event.get("usageMetadata") else None

    def error_body(self, message: str, code: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        return {"error": {"code": 403, "status": "PERMISSION_DENIED",
                          "message": message, "ironharness": dict(detail, code=code)}}
