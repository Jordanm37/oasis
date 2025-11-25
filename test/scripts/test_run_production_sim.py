from __future__ import annotations

import importlib


def test_build_imputer_llm_settings_supports_groq(monkeypatch):
    module = importlib.import_module("scripts.run_production_sim")
    monkeypatch.setattr(module, "IMPUTATION_PROVIDER", "groq")
    monkeypatch.setattr(module, "IMPUTATION_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    settings = module._build_imputer_llm_settings()
    assert settings.provider == "groq"
    assert settings.model_name == "llama-3.3-70b-versatile"
    assert settings.api_key == "groq-key"
    assert settings.base_url == module.API_ENDPOINTS["groq"]

