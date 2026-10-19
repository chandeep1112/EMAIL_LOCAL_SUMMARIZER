from typing import Dict, Optional

from .base import Provider, Ref, read, write
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

PROVIDERS: Dict[str, Provider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "gemini": GeminiProvider(),
}


def get(name: str) -> Optional[Provider]:
    return PROVIDERS.get(name)


def for_path(path: str) -> Optional[str]:
    """Guess the wire format from the request path, for callers that did not
    pick an upstream explicitly (/anthropic/v1/messages and friends)."""
    if path.startswith("/v1beta/"):
        return "gemini"
    if path.startswith("/v1/messages") or path.startswith("/v1/complete"):
        return "anthropic"
    if path.startswith("/v1/"):
        return "openai"
    return None


__all__ = ["Provider", "Ref", "read", "write", "PROVIDERS", "get", "for_path"]
