"""Async background LLM imputer that replaces <LBL:...> placeholders."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from camel.types import ChatCompletion

from configs.llm_settings import (
    IMPUTATION_MAX_TOKENS,
    IMPUTATION_MODEL,
    IMPUTATION_PROVIDER,
    IMPUTATION_TEMPERATURE,
    RAG_IMPUTER_BATCH_SIZE,
    RAG_IMPUTER_RETRIEVAL_TOP_K,
    RAG_IMPUTER_MAX_WORKERS,
    RAG_IMPUTER_STATIC_BANK,
)
from generation.labeler import assign_labels
from oasis.imputation.utils import (
    LABEL_TOKEN_PATTERN,
    StaticBank,
    extract_label_tokens,
)
from orchestrator.model_provider import LLMProviderSettings, create_model_backend

logger = logging.getLogger(__name__)

RagImputerMode = Literal["off", "background", "sync"]


@dataclass
class RagImputerConfig:
    """Runtime configuration for the background imputer."""

    db_path: Path
    llm_settings: LLMProviderSettings
    mode: RagImputerMode = "background"
    batch_size: int = RAG_IMPUTER_BATCH_SIZE
    max_workers: int = RAG_IMPUTER_MAX_WORKERS
    temperature: float = IMPUTATION_TEMPERATURE
    max_tokens: int = IMPUTATION_MAX_TOKENS
    lexicon_terms: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    static_bank_path: Path = Path(RAG_IMPUTER_STATIC_BANK)
    run_seed: int = 314159

    # Post-imputation obfuscation settings
    enable_obfuscation: bool = False  # Set True to obfuscate after imputation
    obfuscation_deterministic: bool = False  # Keep False for realistic variety

    # Vector DB retrieval settings (NEW)
    enable_vector_retrieval: bool = True  # Enable ChromaDB/TF-IDF retrieval
    tfidf_index_dir: Path = Path("data/imputation/tfidf")
    chromadb_index_dir: Path = Path("data/imputation/chromadb")
    retrieval_top_k: int = RAG_IMPUTER_RETRIEVAL_TOP_K  # From configs/llm_settings.py
    use_chromadb: bool = True  # If False, fall back to TF-IDF only


@dataclass
class _ImputeRecord:
    """Item queued for LLM rewrite."""

    table: Literal["post", "comment"]
    record_id: int
    text: str
    tokens: List[str]


class RagImputer:
    """Coordinates async LLM calls to replace label placeholders."""

    def __init__(self, config: RagImputerConfig) -> None:
        self._cfg = config
        self._db_path = config.db_path
        self._mode: RagImputerMode = config.mode
        self._lexicon_terms = {
            str(label): {
                "required": [str(x) for x in (terms.get("required") or [])],
                "optional": [str(x) for x in (terms.get("optional") or [])],
            }
            for label, terms in (config.lexicon_terms or {}).items()
        }
        queue_size = max(1, config.batch_size * max(1, config.max_workers) * 4)
        self._queue: asyncio.Queue[Optional[_ImputeRecord]] = asyncio.Queue(queue_size)
        self._workers: List[asyncio.Task] = []
        self._static_bank = StaticBank({})
        self._last_post_id = 0
        self._last_comment_id = 0
        self._scan_limit = queue_size
        self._started = False

        # Vector retrieval storage
        self._tfidf_indices: Dict[str, Dict[str, Any]] = {}  # dataset_id -> {vectorizer, matrix, texts, labels}
        self._chromadb_client = None
        self._chromadb_collections: Dict[str, Any] = {}
        self._embedding_model = None

    @property
    def mode(self) -> RagImputerMode:
        """Expose configured mode."""
        return self._mode

    async def start(self) -> None:
        """Initialize DB state and spawn workers."""
        if self._mode == "off" or self._started:
            return
        await asyncio.to_thread(self._ensure_columns)
        self._last_post_id = await asyncio.to_thread(self._get_max_id, "post", "post_id")
        self._last_comment_id = await asyncio.to_thread(
            self._get_max_id, "comment", "comment_id"
        )
        bank_path = self._cfg.static_bank_path
        if bank_path and bank_path.exists():
            self._static_bank = StaticBank.load_simple_yaml(bank_path)
        else:
            self._static_bank = StaticBank({})

        # Load vector retrieval indices
        if self._cfg.enable_vector_retrieval:
            await asyncio.to_thread(self._load_vector_indices)

        for idx in range(max(1, self._cfg.max_workers)):
            self._workers.append(asyncio.create_task(self._worker_loop(idx)))
        self._started = True
        logger.info(
            "Started RAG imputer with %s workers (mode=%s, vector_retrieval=%s)",
            len(self._workers), self._mode, self._cfg.enable_vector_retrieval
        )

    async def enqueue_new_rows(self) -> None:
        """Scan DB for newly created posts/comments containing label tokens."""
        if self._mode == "off":
            return
        posts = await asyncio.to_thread(
            self._fetch_rows, "post", "post_id", self._last_post_id
        )
        if posts:
            self._last_post_id = posts[-1].record_id
            for rec in posts:
                await self._queue.put(rec)
        comments = await asyncio.to_thread(
            self._fetch_rows, "comment", "comment_id", self._last_comment_id
        )
        if comments:
            self._last_comment_id = comments[-1].record_id
            for rec in comments:
                await self._queue.put(rec)

    async def flush(self) -> None:
        """Wait until the queue drains."""
        if self._mode == "off":
            return
        await self._queue.join()

    async def shutdown(self) -> None:
        """Stop workers gracefully."""
        if self._mode == "off":
            return
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def _worker_loop(self, idx: int) -> None:
        """Consume queue items and run LLM rewrites."""
        backend = create_model_backend(self._cfg.llm_settings)
        backend.model_config_dict["temperature"] = float(self._cfg.temperature)
        backend.model_config_dict["max_tokens"] = int(self._cfg.max_tokens)
        while True:
            record = await self._queue.get()
            if record is None:
                self._queue.task_done()
                break
            try:
                await self._process_record(record, backend)
            except Exception as exc:  # noqa: BLE001
                logger.exception("RAG imputer worker %s failed: %s", idx, exc)
            finally:
                self._queue.task_done()

    async def _process_record(self, record: _ImputeRecord, backend) -> None:
        """Call the LLM for each placeholder and persist the rewritten text."""
        replacements: List[str] = []
        for occurrence_index, token in enumerate(record.tokens):
            hints = self._build_token_hints(token)
            replacement = await self._generate_replacement(
                backend, record.text, token, occurrence_index, hints
            )
            if not replacement:
                replacement = self._static_bank.deterministic_choice(
                    token, self._cfg.run_seed, record.record_id, occurrence_index
                )
            replacements.append(replacement)
        if not replacements:
            return
        new_text = self._apply_replacements(record.text, replacements)
        
        # Apply post-imputation obfuscation if enabled
        if self._cfg.enable_obfuscation:
            new_text = await self._apply_obfuscation(new_text, record)
        
        await asyncio.to_thread(self._persist_imputed_text, record.table, record.record_id, new_text)
    
    async def _apply_obfuscation(self, text: str, record: _ImputeRecord) -> str:
        """Apply obfuscation to imputed text, targeting harmful terms.
        
        This is called AFTER imputation, so we're obfuscating the actual
        harmful content (slurs, threats, etc.) rather than neutral words.
        """
        try:
            from generation.post_imputation_obfuscator import obfuscate_imputed_text
            
            # Get archetype from the first token (they should all be same archetype)
            archetype = "unknown"
            if record.tokens:
                # Extract archetype from token like "LBL:INCEL_SLANG"
                token = record.tokens[0]
                if "INCEL" in token.upper():
                    archetype = "incel_misogyny"
                elif "HATE" in token.upper() or "SLUR" in token.upper():
                    archetype = "hate_speech"
                elif "THREAT" in token.upper() or "VIOLENT" in token.upper():
                    archetype = "extremist"
                elif "ED_" in token.upper() or "MEANSPO" in token.upper():
                    archetype = "pro_ana"
                elif "BULLY" in token.upper() or "ATTACK" in token.upper():
                    archetype = "bullying"
            
            obfuscated, _ = obfuscate_imputed_text(
                text=text,
                archetype=archetype,
                trajectory_stage="active",  # Default; could be passed in record
                deterministic=self._cfg.obfuscation_deterministic,
            )
            return obfuscated
        except Exception as exc:
            logger.warning("Obfuscation failed: %s", exc)
            return text

    async def _generate_replacement(
        self,
        backend,
        text: str,
        token: str,
        occurrence_index: int,
        hints: List[Dict[str, Any]],
    ) -> str:
        """Invoke the LLM to replace a single placeholder."""
        # Retrieve similar examples from vector DB for few-shot context
        few_shot_examples = ""
        if self._cfg.enable_vector_retrieval and (self._tfidf_indices or self._chromadb_collections):
            try:
                retrieved = self._search_vector_db(
                    query=text,
                    token=token,
                    top_k=self._cfg.retrieval_top_k
                )
                if retrieved:
                    few_shot_examples = self._build_few_shot_examples(retrieved)
            except Exception as e:
                logger.debug(f"Vector retrieval failed: {e}")

        prompt = self._build_user_prompt(text, token, occurrence_index, hints, few_shot_examples)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response: ChatCompletion = await backend.arun(messages)
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            content = ""
            if isinstance(message, dict):
                content = str(message.get("content", "") or "")
            else:
                content = str(getattr(message, "content", "") or "")
            return self._parse_replacement(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM replacement failed for token %s: %s", token, exc)
            return ""

    def _parse_replacement(self, content: str) -> str:
        """Best-effort JSON parsing with string fallback."""
        text = (content or "").strip()
        if not text:
            return ""
        candidate: Optional[str] = None
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ("replacement", "text", "final_text"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        candidate = value.strip()
                        break
        except json.JSONDecodeError:
            candidate = None
        if candidate:
            return self._sanitize(candidate)
        # Fallback to first line when the model ignored JSON instruction.
        first_line = text.splitlines()[0]
        return self._sanitize(first_line)

    def _sanitize(self, value: str) -> str:
        """Clamp whitespace and strip residual label markers."""
        cleaned = value.replace("\n", " ").strip()
        if LABEL_TOKEN_PATTERN.search(cleaned):
            # Avoid leaving unresolved placeholders in the output
            cleaned = LABEL_TOKEN_PATTERN.sub("", cleaned).strip()
        return cleaned[:240]

    def _build_user_prompt(
        self,
        text: str,
        token: str,
        occurrence_index: int,
        hints: List[Dict[str, Any]],
        few_shot_examples: str = "",
    ) -> str:
        """Format the instruction payload for the LLM."""
        hint_lines: List[str] = []
        for hint in hints:
            category = str(hint.get("category", "unknown"))
            req = [str(x) for x in hint.get("required", []) if str(x)]
            opt = [str(x) for x in hint.get("optional", []) if str(x)]
            pieces = [f"category={category}"]
            if req:
                pieces.append(f"required={', '.join(req)}")
            if opt:
                pieces.append(f"optional={', '.join(opt)}")
            hint_lines.append(" - " + "; ".join(pieces))
        hints_block = "\n".join(hint_lines) if hint_lines else " - No lexicon hints available."

        # Include few-shot examples if available (from vector retrieval)
        examples_block = ""
        if few_shot_examples:
            examples_block = f"\n{few_shot_examples}\n\n"

        return (
            "Original post (do not rewrite unrelated text):\n"
            f"{text}\n\n"
            f"Placeholder token #{occurrence_index + 1}: {token}\n"
            "Token guidance:\n"
            f"{hints_block}\n"
            f"{examples_block}"
            "Return JSON of the form {\"replacement\": \"...\"} where the replacement is a short span "
            "that captures the label's intent in the author's voice."
        )

    def _build_token_hints(self, token: str) -> List[Dict[str, Any]]:
        """Map token -> downstream lexicon hints."""
        categories = assign_labels([token])
        hints: List[Dict[str, List[str]]] = []
        for category in categories:
            lex = self._lexicon_terms.get(category) or {}
            hints.append(
                {
                    "category": category,
                    "required": [str(x) for x in lex.get("required", []) if str(x)],
                    "optional": [str(x) for x in lex.get("optional", []) if str(x)],
                }
            )
        return hints

    def _apply_replacements(self, text: str, replacements: List[str]) -> str:
        """Replace tokens in-order with generated spans."""
        iterator = iter(replacements)

        def replacer(match):
            try:
                return next(iterator)
            except StopIteration:
                return match.group(0)

        return LABEL_TOKEN_PATTERN.sub(replacer, text)

    def _persist_imputed_text(self, table: Literal["post", "comment"], record_id: int, text: str) -> None:
        """Write the imputed text back into SQLite."""
        id_column = "post_id" if table == "post" else "comment_id"
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"UPDATE {table} SET text_rag_imputed=? WHERE {id_column}=?",
                (text, record_id),
            )
            conn.commit()

    def _fetch_rows(
        self, table: Literal["post", "comment"], id_column: str, last_id: int
    ) -> List[_ImputeRecord]:
        """Read newly inserted rows that still contain placeholders."""
        # Match both <LBL:...> and bare LBL:... formats
        query = f"""
            SELECT {id_column} AS rid, content
            FROM {table}
            WHERE {id_column} > ?
              AND (content LIKE '%<LBL:%' OR content LIKE '%LBL:%')
              AND (text_rag_imputed IS NULL OR text_rag_imputed = '')
            ORDER BY {id_column} ASC
            LIMIT ?
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (last_id, self._scan_limit)).fetchall()
        records: List[_ImputeRecord] = []
        for row in rows:
            text = row["content"] or ""
            tokens = extract_label_tokens(text)
            if not tokens:
                continue
            records.append(
                _ImputeRecord(
                    table=table,
                    record_id=int(row["rid"]),
                    text=text,
                    tokens=tokens,
                )
            )
        return records

    def _ensure_columns(self) -> None:
        """Add text_rag_imputed columns where needed."""
        with sqlite3.connect(self._db_path) as conn:
            for table in ("post", "comment"):
                cols = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if "text_rag_imputed" not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN text_rag_imputed TEXT")
            conn.commit()

    def _get_max_id(self, table: str, column: str) -> int:
        """Get the current max primary key value."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
            return int(row[0] or 0)

    def _load_vector_indices(self) -> None:
        """Load TF-IDF and optionally ChromaDB indices for vector retrieval."""
        # Load TF-IDF indices
        tfidf_dir = self._cfg.tfidf_index_dir
        if tfidf_dir.exists():
            for pkl_file in tfidf_dir.glob("*.pkl"):
                dataset_id = pkl_file.stem
                try:
                    with open(pkl_file, "rb") as f:
                        data = pickle.load(f)
                    self._tfidf_indices[dataset_id] = data
                    logger.info(f"Loaded TF-IDF index: {dataset_id} ({data.get('num_documents', 'N/A')} docs)")
                except Exception as e:
                    logger.warning(f"Failed to load TF-IDF index {dataset_id}: {e}")

        # Load ChromaDB if enabled
        if self._cfg.use_chromadb and self._cfg.chromadb_index_dir.exists():
            try:
                import chromadb
                from chromadb.config import Settings

                # Load embedding model
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

                # Load each dataset's collection
                for dataset_dir in self._cfg.chromadb_index_dir.iterdir():
                    if dataset_dir.is_dir():
                        dataset_id = dataset_dir.name
                        try:
                            client = chromadb.PersistentClient(
                                path=str(dataset_dir),
                                settings=Settings(anonymized_telemetry=False)
                            )
                            collection = client.get_collection(dataset_id)
                            self._chromadb_collections[dataset_id] = collection
                            logger.info(f"Loaded ChromaDB collection: {dataset_id} ({collection.count()} docs)")
                        except Exception as e:
                            logger.warning(f"Failed to load ChromaDB collection {dataset_id}: {e}")
            except ImportError:
                logger.warning("ChromaDB/sentence-transformers not installed, using TF-IDF only")
                self._cfg.use_chromadb = False

        total_tfidf = len(self._tfidf_indices)
        total_chroma = len(self._chromadb_collections)
        logger.info(f"Vector retrieval ready: {total_tfidf} TF-IDF, {total_chroma} ChromaDB indices")

    def _search_vector_db(
        self,
        query: str,
        token: str,
        top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """Search vector indices for similar content.

        Args:
            query: The query text (post content)
            token: The label token (e.g., "LBL:HATE_SPEECH")
            top_k: Number of results to return

        Returns:
            List of (text, label, score) tuples
        """
        results = []

        # Build search query from post context + token hint
        token_hint = token.replace("LBL:", "").replace("_", " ").lower()
        search_query = f"{query} {token_hint}"

        # Try ChromaDB first (semantic search)
        if self._cfg.use_chromadb and self._chromadb_collections and self._embedding_model:
            try:
                query_embedding = self._embedding_model.encode([search_query])[0].tolist()

                for dataset_id, collection in self._chromadb_collections.items():
                    try:
                        search_results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=top_k
                        )
                        if search_results["documents"] and search_results["documents"][0]:
                            for i, doc in enumerate(search_results["documents"][0]):
                                label = search_results["metadatas"][0][i].get("label", "unknown") if search_results["metadatas"] else "unknown"
                                distance = search_results["distances"][0][i] if search_results["distances"] else 1.0
                                score = 1.0 - distance  # Convert distance to similarity
                                results.append((doc, label, score))
                    except Exception as e:
                        logger.debug(f"ChromaDB search failed for {dataset_id}: {e}")
            except Exception as e:
                logger.warning(f"ChromaDB search error: {e}")

        # Fall back to TF-IDF if no ChromaDB results
        if not results and self._tfidf_indices:
            for dataset_id, index_data in self._tfidf_indices.items():
                try:
                    vectorizer = index_data["vectorizer"]
                    matrix = index_data["matrix"]
                    texts = index_data["texts"]
                    labels = index_data.get("labels", ["unknown"] * len(texts))

                    query_vec = vectorizer.transform([search_query])
                    similarities = (matrix * query_vec.T).toarray().flatten()

                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    for idx in top_indices:
                        if similarities[idx] > 0.05:  # Min threshold
                            results.append((texts[idx], labels[idx], float(similarities[idx])))
                except Exception as e:
                    logger.debug(f"TF-IDF search failed for {dataset_id}: {e}")

        # Sort by score and return top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def _build_few_shot_examples(self, retrieved: List[Tuple[str, str, float]]) -> str:
        """Build few-shot examples from retrieved content.

        Args:
            retrieved: List of (text, label, score) tuples

        Returns:
            Formatted examples string
        """
        if not retrieved:
            return ""

        examples = []
        for text, label, score in retrieved[:3]:
            # Truncate long texts
            text_preview = text[:150] + "..." if len(text) > 150 else text
            examples.append(f"- [{label}] \"{text_preview}\"")

        return "Similar examples from real data:\n" + "\n".join(examples)


_SYSTEM_PROMPT = (
    "You replace placeholder label tokens (e.g., <LBL:INCEL_MISOGYNY>) "
    "inside social-media posts. Only replace the placeholder spans, keep the author's "
    "voice, and keep the rest of the text identical. Output JSON."
)

