from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from oasis.imputation.rag_llm_imputer import RagImputer, RagImputerConfig
from oasis.imputation.utils import extract_label_tokens
from orchestrator.model_provider import LLMProviderSettings


def _init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE post (post_id INTEGER PRIMARY KEY, content TEXT, text_rag_imputed TEXT)"
    )
    conn.execute(
        "CREATE TABLE comment (comment_id INTEGER PRIMARY KEY, post_id INTEGER, content TEXT, text_rag_imputed TEXT)"
    )
    conn.commit()
    conn.close()


def _read_column(db_path: Path, table: str) -> str:
    conn = sqlite3.connect(db_path)
    column = conn.execute(f"SELECT text_rag_imputed FROM {table} WHERE {table}_id=1").fetchone()[0]
    conn.close()
    return column or ""


@pytest.mark.asyncio
async def test_rag_imputer_replaces_tokens(tmp_path, monkeypatch):
    db_path = tmp_path / "sim.db"
    _init_db(db_path)
    bank_path = tmp_path / "bank.yaml"
    bank_path.write_text("LBL:INCEL_SLANG:\n  - fallback\nLBL:SUPPORTIVE:\n  - cheer\n", encoding="utf-8")

    async def fake_worker_loop(self, idx):
        while True:
            record = await self._queue.get()
            if record is None:
                self._queue.task_done()
                break
            await RagImputer._process_record(self, record, backend=None)
            self._queue.task_done()

    async def fake_generate(self, backend, text, token, occurrence_index, hints):
        return f"[{token.lower()}-{occurrence_index}]"

    monkeypatch.setattr(RagImputer, "_worker_loop", fake_worker_loop, raising=False)
    monkeypatch.setattr(RagImputer, "_generate_replacement", fake_generate, raising=False)

    cfg = RagImputerConfig(
        db_path=db_path,
        llm_settings=LLMProviderSettings(
            provider="xai",
            model_name="stub",
            api_key="test",
            base_url="https://example.com",
        ),
        mode="sync",
        batch_size=2,
        max_workers=1,
        static_bank_path=bank_path,
        run_seed=42,
    )
    imputer = RagImputer(cfg)
    await imputer.start()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO post (post_id, content) VALUES (1, 'Hello <LBL:INCEL_SLANG> world')"
    )
    conn.execute(
        "INSERT INTO comment (comment_id, post_id, content) VALUES (1, 1, 'Reply <LBL:SUPPORTIVE> here')"
    )
    conn.commit()
    conn.close()
    await imputer.enqueue_new_rows()
    await imputer.flush()
    await imputer.shutdown()

    post_text = _read_column(db_path, "post")
    comment_text = _read_column(db_path, "comment")
    assert "[lbl:incel_slang-0]" in post_text
    assert "[lbl:supportive-0]" in comment_text


def test_extract_label_tokens_handles_multiple():
    text = "A <LBL:INCEL_SLANG> and <LBL:MISINFO_CLAIM> example."
    tokens = extract_label_tokens(text)
    assert tokens == ["LBL:INCEL_SLANG", "LBL:MISINFO_CLAIM"]

