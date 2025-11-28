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
from configs.adaptive_rate_limiter import AdaptiveRateLimiter
from generation.labeler import assign_labels
from oasis.imputation.utils import (
    LABEL_TOKEN_PATTERN,
    StaticBank,
    extract_label_tokens,
)
from orchestrator.model_provider import LLMProviderSettings, create_model_backend

logger = logging.getLogger(__name__)

RagImputerMode = Literal["off", "background", "sync"]

# Map variant label names to canonical labels for lexicon lookup.
# This handles cases where persona generator emits label names that need
# to be mapped to entries that exist in the static bank or lexicons.
# The imputer uses these to provide better hints to the LLM.
LABEL_ALIASES: Dict[str, str] = {
    # === Recovery mappings ===
    "RECOVERY_ED": "RECOVERY_SUPPORT",
    "SH_METHOD": "SELF_HARM",

    # === Conspiracy mappings ===
    "CONSPIRACY": "CONSPIRACY_NARRATIVE",
    "COVER_UP": "CONSPIRACY_NARRATIVE",
    "CONTROLLED_OPP": "CONSPIRACY_NARRATIVE",
    "WAKE_UP": "MISINFO_CLAIM",

    # === ED/Pro-ana mappings ===
    "ED_RISK": "ED_PROMO",
    "SELF_HARM": "ED_PROMO",
    "FASTING_GLORIFY": "ED_PROMO",
    "CALORIE_OBSESSION": "ED_METHOD",
    "RESTRICTION_TIP": "ED_METHOD",

    # === Incel/misogyny mappings ===
    "MISOGYNY": "INCEL_MISOGYNY",
    "BLACKPILL": "INCEL_MISOGYNY",
    "FOID_SLUR": "INCEL_MISOGYNY",
    "REDPILL_WISDOM": "MISOGYNISTIC_LECTURE",
    "FRAME_CONTROL": "MISOGYNISTIC_LECTURE",
    "SMV_THEORY": "OBJECTIFICATION",
    "LOOKISM": "OBJECTIFICATION",

    # === Hate speech mappings ===
    "RACIAL_SLUR": "HATE_SLUR",
    "SLUR": "HATE_SLUR",
    "SUPREMACIST": "HATE_SLUR",
    "VERMIN_RHETORIC": "DEHUMANIZATION",
    "ETHNIC_ATTACK": "ETHNIC_TARGETING",
}


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

    # Persona context settings
    enable_persona_context: bool = True  # Include author persona in prompt
    personas_csv_path: Optional[Path] = None  # Path to personas CSV file


@dataclass
class _ImputeRecord:
    """Item queued for LLM rewrite."""

    table: Literal["post", "comment"]
    record_id: int
    text: str
    tokens: List[str]
    user_id: Optional[int] = None  # For persona context lookup


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

        # Persona context storage (user_id -> persona dict)
        self._personas: Dict[int, Dict[str, str]] = {}
        
        # Adaptive rate limiter for imputation (TPM-aware)
        # llama-3.3-70b has 300K TPM, ~500 tokens/request = 600 RPM effective
        self._rate_limiter = AdaptiveRateLimiter(
            model="llama-3.3-70b-versatile",
            initial_concurrency=min(config.max_workers, 10),
            max_concurrency=config.max_workers,
            min_concurrency=2,
            tpm_limit=300_000,
        )

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

        # Load persona context if enabled
        if self._cfg.enable_persona_context and self._cfg.personas_csv_path:
            await asyncio.to_thread(self._load_personas)

        for idx in range(max(1, self._cfg.max_workers)):
            self._workers.append(asyncio.create_task(self._worker_loop(idx)))
        self._started = True
        logger.info(
            "Started RAG imputer with %s workers (mode=%s, vector_retrieval=%s, personas=%s)",
            len(self._workers), self._mode, self._cfg.enable_vector_retrieval, len(self._personas)
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
                backend, record.text, token, occurrence_index, hints, record.user_id
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
        user_id: Optional[int] = None,
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

        prompt = self._build_user_prompt(text, token, occurrence_index, hints, few_shot_examples, user_id)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        # Retry with exponential backoff for rate limits (up to 5 retries)
        max_retries = 5
        base_delay = 0.2  # Start with 200ms
        
        for attempt in range(max_retries):
            try:
                # Use adaptive rate limiter to manage TPM
                async with await self._rate_limiter.acquire():
                    response: ChatCompletion = await backend.arun(messages)
                
                # Record success for rate limiter adaptation
                self._rate_limiter.record_success()
                
                choice = response.choices[0]
                message = getattr(choice, "message", None)
                content = ""
                if isinstance(message, dict):
                    content = str(message.get("content", "") or "")
                else:
                    content = str(getattr(message, "content", "") or "")
                return self._parse_replacement(content)
            except Exception as exc:  # noqa: BLE001
                error_str = str(exc).lower()
                is_rate_limit = "429" in error_str or "rate limit" in error_str or "too many" in error_str
                
                if is_rate_limit:
                    # Extract retry-after if present
                    import re
                    retry_after_ms = None
                    retry_match = re.search(r'try again in (\d+(?:\.\d+)?)\s*(?:ms|s)', error_str)
                    if retry_match:
                        retry_val = float(retry_match.group(1))
                        if 'ms' in error_str[retry_match.start():retry_match.end()+5]:
                            retry_after_ms = int(retry_val)
                        else:
                            retry_after_ms = int(retry_val * 1000)
                    
                    # Record rate limit for adaptive adjustment
                    self._rate_limiter.record_rate_limit(retry_after_ms=retry_after_ms)
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + (asyncio.get_event_loop().time() % 0.1)
                        if retry_after_ms:
                            delay = max(delay, retry_after_ms / 1000.0 + 0.05)
                        
                        logger.debug(f"Rate limited on {token}, retry {attempt+1}/{max_retries} after {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue
                
                logger.warning("LLM replacement failed for token %s: %s", token, exc)
                return ""
        
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
        user_id: Optional[int] = None,
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

        # Include persona context if available
        persona_block = ""
        if self._cfg.enable_persona_context and user_id and user_id in self._personas:
            persona = self._personas[user_id]
            persona_block = (
                "\n\nAuthor persona:\n"
                f"- Username: {persona.get('username', 'unknown')}\n"
                f"- Type: {persona.get('primary_label', 'unknown')}\n"
                f"- Bio: {persona.get('bio', '')[:150]}\n"
            )

        return (
            "Original post (do not rewrite unrelated text):\n"
            f"{text}"
            f"{persona_block}\n\n"
            f"Placeholder token #{occurrence_index + 1}: {token}\n"
            "Token guidance:\n"
            f"{hints_block}\n"
            f"{examples_block}"
            'Return JSON {"replacement": "..."} with a phrase (2-10 words) that:\n'
            "- Sounds like something THIS USER would actually say based on their persona\n"
            "- Fits naturally in the sentence\n"
            "- Matches the INTENSITY of the post\n"
            "- Expresses the label's sentiment authentically"
        )

    def _build_token_hints(self, token: str) -> List[Dict[str, Any]]:
        """Map token -> downstream lexicon hints.

        Applies label alias mapping to handle variant label names
        (e.g., CONSPIRACY -> CONSPIRACY_NARRATIVE) before lexicon lookup.
        """
        # Extract the label name from the token (strip LBL: prefix and angle brackets)
        label_name = token.replace("<", "").replace(">", "").replace("LBL:", "").strip()

        # Apply alias mapping if the label has a canonical form
        canonical_label = LABEL_ALIASES.get(label_name, label_name)

        # Create the normalized token for assign_labels
        normalized_token = f"LBL:{canonical_label}"

        categories = assign_labels([normalized_token])
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

        # Log if we applied an alias mapping
        if canonical_label != label_name:
            logger.debug(f"Label alias applied: {label_name} -> {canonical_label}")

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
            SELECT {id_column} AS rid, user_id, content
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
                    user_id=int(row["user_id"]) if row["user_id"] else None,
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

    def _load_personas(self) -> None:
        """Load persona data from CSV and map to user_ids via database."""
        import csv

        personas_path = self._cfg.personas_csv_path
        if not personas_path or not personas_path.exists():
            logger.warning(f"Personas CSV not found: {personas_path}")
            return

        # Load personas by username
        personas_by_username: Dict[str, Dict[str, str]] = {}
        try:
            with open(personas_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    username = row.get("username", "")
                    if username:
                        personas_by_username[username] = dict(row)
        except Exception as e:
            logger.warning(f"Failed to load personas CSV: {e}")
            return

        # Map user_id -> persona via username from database
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT user_id, name FROM user").fetchall()
            for row in rows:
                username = row["name"]
                user_id = row["user_id"]
                if username in personas_by_username:
                    self._personas[user_id] = personas_by_username[username]

        logger.info(f"Loaded {len(self._personas)} persona mappings")

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
    "You replace placeholder label tokens inside social-media posts. "
    "You will be given the author's persona/profile to help you stay in character. "
    "Rules: (1) Output 2-10 words, (2) STAY IN CHARACTER as the post author - never sound educational, "
    "(3) Never include the token text itself, (4) MATCH the toxicity/intensity level of the surrounding text, "
    '(5) Output JSON {"replacement": "..."}.'
)

