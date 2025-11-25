from __future__ import annotations

from typing import Any, Dict

import pytest

from oasis.generation import groq_client
from oasis.generation.groq_client import GroqConfig, generate_text


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - no-op
        return

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_generate_text_rest_fallback(monkeypatch):
    monkeypatch.setattr(groq_client, "_HAS_GROQ_SDK", False)

    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["timeout"] = timeout
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "answer from groq",
                    }
                }
            ]
        }
        return _FakeResponse(payload)

    monkeypatch.setattr(groq_client.requests, "post", fake_post)

    cfg = GroqConfig(
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        top_p=0.8,
        max_tokens=256,
        base_url="https://api.groq.com/openai/v1",
        timeout_seconds=30.0,
    )
    out = generate_text("system", "user text", cfg)
    assert out == "answer from groq"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert "Bearer test-key" in captured["headers"]["Authorization"]
    assert '"model": "llama-3.3-70b-versatile"' in captured["data"]


def test_generate_text_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        generate_text("system", "user")

