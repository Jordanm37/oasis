"""LLM generation client helpers."""

import importlib

gemini_client = importlib.import_module(".gemini_client", __name__)
groq_client = importlib.import_module(".groq_client", __name__)
xai_client = importlib.import_module(".xai_client", __name__)

__all__ = ["gemini_client", "groq_client", "xai_client"]
