from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq as GroqClient  # type: ignore

    _HAS_GROQ_SDK = True
except Exception:  # pragma: no cover
    GroqClient = None
    _HAS_GROQ_SDK = False

import requests


@dataclass
class GroqConfig:
    r"""Configuration for Groq chat completions."""

    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 2048
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_seconds: float = 120.0


def generate_text(
    system_instruction: str,
    user_text: str,
    config: Optional[GroqConfig] = None,
) -> str:
    r"""Generate text using Groq's OpenAI-compatible API (SDK preferred)."""
    api_key = (config.api_key if config else os.getenv("GROQ_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = (config.model if config else os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    temperature = config.temperature if config else 0.4
    top_p = config.top_p if config else 0.95
    max_tokens = config.max_tokens if config else 2048
    base_url = config.base_url if config else os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    timeout = config.timeout_seconds if config else 120.0

    if _HAS_GROQ_SDK:
        try:
            client = GroqClient(api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_tokens,
            )
            message = completion.choices[0].message  # type: ignore[index]
            content = getattr(message, "content", "") or ""
            return str(content).strip()
        except Exception:
            pass

    try:
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = (choices[0] or {}).get("message", {})
        return str(message.get("content", "") or "").strip()
    except Exception:
        return ""

