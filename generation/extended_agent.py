from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from camel.messages import BaseMessage
from tenacity import (retry, retry_if_exception, stop_after_attempt,
                      wait_random_exponential)

from configs.llm_settings import (DEFAULT_RATE_LIMITS, RATE_LIMITS,
                                  SIMULATION_MAX_TOKENS, get_rate_limits)
from generation.emission_policy import EmissionPolicy, PersonaConfig
from generation.labeler import assign_labels
from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.typing import ActionType
from orchestrator.expect_registry import ExpectRegistry
from orchestrator.llm_config import LLM_CONFIG
from orchestrator.rng import DeterministicRNG
from orchestrator.scheduler import MultiLabelScheduler
from orchestrator.sidecar_logger import SidecarLogger

_TOKEN_RE = re.compile(r"<LBL:[A-Z_]+>")


class _TokenBucketLimiter:
    """Async token-bucket limiter for RPM, TPM and RPS."""

    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
        enabled: bool = True,
        rps: int | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            rpm: Requests per minute (overrides model config if provided)
            tpm: Tokens per minute (overrides model config if provided)
            enabled: Whether rate limiting is enabled
            rps: Requests per second (overrides model config if provided)
            model: Model name to look up rate limits from RATE_LIMITS config
        """
        self.enabled = bool(enabled)
        now = time.monotonic()

        # Get rate limits from config if model specified
        limits = get_rate_limits(model) if model else DEFAULT_RATE_LIMITS

        # Use provided values or fall back to config
        rpm_val = rpm if rpm is not None else limits.get("rpm", DEFAULT_RATE_LIMITS["rpm"])
        tpm_val = tpm if tpm is not None else limits.get("tpm", DEFAULT_RATE_LIMITS["tpm"])
        rps_val = rps if rps is not None else limits.get("rps", DEFAULT_RATE_LIMITS["rps"])
        self._retries = limits.get("retries", DEFAULT_RATE_LIMITS["retries"])

        # Requests per minute bucket
        self._rpm_capacity = float(max(1, rpm_val))
        self._rpm_tokens = float(self._rpm_capacity)
        self._rpm_refill_rate = self._rpm_capacity / 60.0  # tokens per second
        self._rpm_last = now
        # Tokens per minute bucket
        self._tpm_capacity = float(max(1, tpm_val))
        self._tpm_tokens = float(self._tpm_capacity)
        self._tpm_refill_rate = self._tpm_capacity / 60.0
        self._tpm_last = now
        # Requests per second bucket
        self._rps_capacity = float(max(1, rps_val))
        self._rps_tokens = float(self._rps_capacity)
        self._rps_refill_rate = self._rps_capacity / 1.0
        self._rps_last = now
        # Sync
        self._lock = asyncio.Lock()

    def _refill_unlocked(self) -> None:
        now = time.monotonic()
        # RPM
        elapsed = max(0.0, now - self._rpm_last)
        self._rpm_tokens = min(
            self._rpm_capacity, self._rpm_tokens + elapsed * self._rpm_refill_rate
        )
        self._rpm_last = now
        # TPM
        elapsed_t = max(0.0, now - self._tpm_last)
        self._tpm_tokens = min(
            self._tpm_capacity, self._tpm_tokens + elapsed_t * self._tpm_refill_rate
        )
        self._tpm_last = now
        # RPS
        elapsed_s = max(0.0, now - self._rps_last)
        self._rps_tokens = min(
            self._rps_capacity, self._rps_tokens + elapsed_s * self._rps_refill_rate
        )
        self._rps_last = now

    async def acquire(self, est_tokens: int = 1024) -> None:
        if not self.enabled:
            return
        est_tokens = int(max(1, est_tokens))
        while True:
            wait_for: float = 0.0
            async with self._lock:
                self._refill_unlocked()
                have_rpm = self._rpm_tokens >= 1.0
                have_tpm = self._tpm_tokens >= float(est_tokens)
                have_rps = self._rps_tokens >= 1.0
                if have_rpm and have_tpm and have_rps:
                    self._rpm_tokens -= 1.0
                    self._tpm_tokens -= float(est_tokens)
                    self._rps_tokens -= 1.0
                    return
                # compute wait time until sufficient tokens
                need_rpm = max(0.0, 1.0 - self._rpm_tokens)
                need_tpm = max(0.0, float(est_tokens) - self._tpm_tokens)
                wait_rpm = need_rpm / self._rpm_refill_rate if need_rpm > 0 else 0.0
                wait_tpm = need_tpm / self._tpm_refill_rate if need_tpm > 0 else 0.0
                need_rps = max(0.0, 1.0 - self._rps_tokens)
                wait_rps = need_rps / self._rps_refill_rate if need_rps > 0 else 0.0
                wait_for = max(wait_rpm, wait_tpm, wait_rps, 0.01)  # min sleep
            await asyncio.sleep(min(wait_for, 2.0))

    @staticmethod
    def estimate_tokens(model: str | None = None) -> int:
        """Estimate tokens for a request.

        Args:
            model: Optional model name to get max_tokens from config
        """
        # Use centralized config for token estimation
        limits = get_rate_limits(model) if model else DEFAULT_RATE_LIMITS
        max_tokens = SIMULATION_MAX_TOKENS
        est_prompt = 12000  # Estimated prompt tokens
        return max(1, est_prompt + max_tokens)


# Centralized LLM config (kept for backwards compatibility)
_CFG = LLM_CONFIG

# Model-specific limiters cache
_MODEL_LIMITERS: Dict[str, _TokenBucketLimiter] = {}


def _get_limiter(model: str) -> _TokenBucketLimiter:
    """Get or create a rate limiter for a specific model."""
    if model not in _MODEL_LIMITERS:
        _MODEL_LIMITERS[model] = _TokenBucketLimiter(
            model=model,
            enabled=bool(_CFG.rate_limit_enabled),
        )
    return _MODEL_LIMITERS[model]


# Default xAI limiter for backwards compatibility
_XAI_LIMITER = _TokenBucketLimiter(
    model="grok-4-fast-non-reasoning",
    enabled=bool(_CFG.rate_limit_enabled),
)


def _should_retry_rate_limit(exc: BaseException) -> bool:
    s = (str(exc) or "").lower()
    return "rate limit" in s or "429" in s or "too many requests" in s


class ExtendedSocialAgent(SocialAgent):
    r"""SocialAgent extension that injects per-step label-token instructions and logs sidecar."""

    def __init__(
        self,
        *args,
        persona_cfg: PersonaConfig,
        emission_policy: EmissionPolicy,
        sidecar_logger: SidecarLogger,
        run_seed: int,
        expect_registry: Optional[ExpectRegistry] = None,
        scheduler: Optional[MultiLabelScheduler] = None,
        harm_priors: Optional[Dict[str, float]] = None,
        guidance_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._persona_cfg: PersonaConfig = persona_cfg
        self._policy: EmissionPolicy = emission_policy
        self._sidecar: SidecarLogger = sidecar_logger
        self._run_seed: int = int(run_seed)
        self._step_index: int = 0
        self._expect_registry: Optional[ExpectRegistry] = expect_registry
        self._scheduler: Optional[MultiLabelScheduler] = scheduler
        self._harm_priors: Dict[str, float] = dict(harm_priors or {})
        self._guidance_config: Dict[str, Any] = dict(guidance_config or {})

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(_CFG.xai_retry_attempts)),
        wait=wait_random_exponential(multiplier=0.5, max=20.0),
        retry=retry_if_exception(_should_retry_rate_limit),
    )
    async def _retryable_super_astep(self, user_msg: BaseMessage):
        return await super().astep(user_msg)

    async def astep(self, user_msg: BaseMessage):
        # Rate-limit before each LLM step using model-specific limiter
        try:
            # Resolve model name robustly whether it's a string or Enum-like
            name_obj = getattr(self, "model_type", None)
            model_name = ""
            if hasattr(name_obj, "value"):
                model_name = str(getattr(name_obj, "value"))
            elif name_obj is not None:
                model_name = str(name_obj)
            # Fallbacks: check common backend attributes
            if not model_name:
                backend = getattr(self, "model_backend", None) or getattr(self, "model", None)
                alt = getattr(backend, "model_type", None)
                if hasattr(alt, "value"):
                    model_name = str(getattr(alt, "value"))
                elif alt is not None:
                    model_name = str(alt)
            # Use model-specific rate limiter from centralized RATE_LIMITS config
            if model_name:
                limiter = _get_limiter(model_name)
                await limiter.acquire(limiter.estimate_tokens(model_name))
        except Exception:
            # Fallback: proceed even if limiter fails
            pass
        return await self._retryable_super_astep(user_msg)

    async def perform_action_by_llm(self):
        # Build step-scoped RNG and emission decision
        # For lack of direct thread_id here, use agent_id as a stable scope.
        user_id = int(self.social_agent_id)
        thread_scope = f"a_{user_id}"
        override = self._scheduler.get_mode_override(self._persona_cfg) if self._scheduler else None
        # Optional dynamic token weighting context derived from harm priors
        context: Optional[Dict[str, Any]] = None
        try:
            if bool(self._guidance_config.get("token_weighting", False)):
                intensity = float(self._guidance_config.get("intensity", 1.0))
                pri = self._harm_priors or {}
                toxicity = float(pri.get("toxicity", 0.0))
                insult = float(pri.get("insult", 0.0))
                profanity = float(pri.get("profanity", 0.0))
                # Token biases (small, additive; label-agnostic)
                dyn: Dict[str, float] = {}
                # Incel space
                dyn["LBL:HARASSMENT"] = max(0.0, 0.6 * (insult + profanity + toxicity) / 3.0 * intensity)
                dyn["LBL:INCEL_SLANG"] = 0.0
                # Misinfo/conspiracy
                dyn["LBL:MISINFO_CLAIM"] = max(0.0, 0.4 * toxicity * intensity)
                dyn["LBL:MISINFO_SOURCE"] = max(0.0, 0.2 * toxicity * intensity)
                # Recovery/benign unchanged
                context = {"dynamic_token_probs": dyn}
        except Exception:
            context = None
        context_extra = {
            "style_variation": self._persona_cfg.style_variation or {},
            "prompt_metadata": self._persona_cfg.prompt_metadata or {},
            "lexicon_samples": self._persona_cfg.lexicon_samples or {},
        }
        decision = self._policy.decide(
            user_id=user_id,
            thread_id=thread_scope,
            step_idx=self._step_index,
            persona=self._persona_cfg,
            context={**(context or {}), **context_extra},
            override_post_mode_probs=override,
        )

        # Record expectation for programmatic fallback
        if self._expect_registry is not None:
            await self._expect_registry.set_expected(user_id, self._step_index, decision.get("tokens", []))

        # Augment user message with a concise step instruction
        style_hint = self._maybe_style_hint(decision)
        step_hint = self._format_step_hint(decision)
        lexical_hint = self._format_lexical_hint(user_id, decision)
        env_prompt = await self.env.to_text_prompt()
        style_prompt_hint = self._format_prompt_goal_hint()
        hint_parts = [part for part in (style_hint, lexical_hint, style_prompt_hint, step_hint) if part]
        hint_prefix = "\n".join(hint_parts)
        if hint_prefix:
            hint_prefix += "\n"
        content_text = f"{hint_prefix}Here is your social media environment: {env_prompt}"
        user_msg = BaseMessage.make_user_message(role_name="User", content=content_text)
        try:
            response = await self.astep(user_msg)
            # Log each tool call to sidecar with detected tokens and labels.
            tool_calls = response.info.get("tool_calls", [])
            for tool_call in tool_calls:
                action_name = tool_call.tool_name
                args = tool_call.args or {}
                result = getattr(tool_call, "result", None)
                await self._log_sidecar(action_name, args, result, decision)
            self._step_index += 1
            return response
        except Exception as e:
            # Keep parity with base class error handling
            from oasis.social_agent.agent import agent_log
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            return e

    def _format_step_hint(self, decision: Dict[str, Any]) -> str:
        mode = decision.get("mode", "none")
        tokens = decision.get("tokens", [])
        if mode == "none":
            return "This is a benign neutral comment. Do not include any label markers."
        if mode == "single" and tokens:
            return f"Use exactly one label marker inline: {tokens[0]}."
        if mode == "double" and len(tokens) >= 2:
            return f"Use exactly two label markers inline: {tokens[0]} and {tokens[1]}."
        # Fallback
        return "If natural, include label markers as instructed; otherwise keep it neutral."

    def _format_lexical_hint(self, user_id: int, decision: Dict[str, Any]) -> str:
        label_samples = decision.get("label_lexicon_samples") or {}
        union_optional: List[str] = []
        for _, terms in label_samples.items():
            opts = (terms or {}).get("optional") or []
            union_optional.extend([str(x) for x in opts])
        if not union_optional:
            samples = self._persona_cfg.lexicon_samples or {}
            sampled = samples.get("sampled") if isinstance(samples, dict) else {}
            union_optional = sampled.get("optional") if isinstance(sampled, dict) else []
        if not union_optional:
            return ""
        rng = DeterministicRNG(self._run_seed).fork("lexicon_hint", f"user:{user_id}", f"step:{self._step_index}")
        choices = []
        pool = list(union_optional)
        while pool and len(choices) < 2:
            weights = {term: 1.0 for term in pool}
            pick = rng.fork(f"choice_{len(choices)}").categorical(weights)
            choices.append(pick)
            pool.remove(pick)
        if not choices:
            return ""
        return f"Lexical flavor: try weaving in words like {', '.join(choices)}."

    def _format_prompt_goal_hint(self) -> str:
        meta = self._persona_cfg.prompt_metadata or {}
        goal = meta.get("persona_goal")
        quirks = meta.get("style_quirks")
        indicators = meta.get("style_indicators")
        snippets: List[str] = []
        if goal:
            snippets.append(f"Goal: {goal}")
        if quirks:
            snippets.append(f"Style quirks: {quirks}")
        if isinstance(indicators, dict) and indicators:
            formatted = "; ".join(f"{k}={v}" for k, v in sorted(indicators.items()))
            snippets.append(f"Style indicators: {formatted}")
        return " ".join(snippets).strip()

    def _apply_post_variations(self, text: str, label_lex_samples: Optional[Dict[str, Any]] = None) -> str:
        user_id = int(self.social_agent_id)
        text = self._apply_lexicon_variation(text, label_lex_samples)
        text = self._apply_style_variation(text, user_id)
        return text

    def _apply_lexicon_variation(self, text: str, label_lex_samples: Optional[Dict[str, Any]] = None) -> str:
        if not text:
            return text
        # Prefer label-level samples if available
        optional: List[str] = []
        required: List[str] = []
        if label_lex_samples:
            for _, terms in (label_lex_samples or {}).items():
                required.extend([str(t) for t in (terms.get("required") or [])])
                optional.extend([str(t) for t in (terms.get("optional") or [])])
        if not optional and not required:
            lex_samples = self._persona_cfg.lexicon_samples or {}
            sampled = lex_samples.get("sampled") if isinstance(lex_samples, dict) else {}
            optional = sampled.get("optional") if isinstance(sampled, dict) else []
            required = sampled.get("required") if isinstance(sampled, dict) else []
        user_id = int(self.social_agent_id)
        rng = DeterministicRNG(self._run_seed).fork("lexicon_apply", f"user:{user_id}", f"step:{self._step_index}")

        # Ensure at least one required term appears if provided
        if required:
            missing = [term for term in required if term and term not in text]
            if missing:
                text = text.rstrip() + (" " if not text.endswith(" ") else "") + missing[0]

        if not optional:
            return text

        pool = [term for term in optional if term]
        chosen: List[str] = []
        while pool and len(chosen) < 2:
            weights = {term: 1.0 for term in pool}
            try:
                pick = rng.fork(f"opt_{len(chosen)}").categorical(weights)
            except Exception:
                break
            chosen.append(pick)
            pool.remove(pick)

        if not chosen:
            return text

        # Insert chosen lexicon near the end of a randomly selected sentence.
        sentences = re.split(r"([.!?])", text)
        if not sentences:
            return text + " " + " ".join(chosen)
        sentence_count = max(1, len(sentences) // 2)
        insert_idx = rng.fork("sentence_pick")._python_random().randint(0, sentence_count - 1)
        insert_pos = min(len(sentences) - 1, insert_idx * 2)
        sentences[insert_pos] = (sentences[insert_pos] or "") + " " + " ".join(chosen)
        return "".join(sentences).strip()

    def _apply_style_variation(self, text: str, user_id: int) -> str:
        variation = self._persona_cfg.style_variation or {}
        if not variation or not text:
            return text
        rng = DeterministicRNG(self._run_seed).fork("style_variation", f"user:{user_id}", f"step:{self._step_index}")
        result = text
        emoji_rate = float(variation.get("emoji_rate", 0.0) or 0.0)
        if emoji_rate > 0.0 and rng.fork("emoji_flag").bernoulli(min(1.0, emoji_rate)):
            emojis = {"🙂": 1.0, "😉": 1.0, "😂": 1.0, "🔥": 1.0, "✨": 1.0}
            emoji = rng.fork("emoji_pick").categorical(emojis)
            result = result.rstrip() + " " + emoji
        typo_rate = float(variation.get("typo_rate", 0.0) or 0.0)
        if typo_rate > 0.0 and rng.fork("typo_flag").bernoulli(min(1.0, typo_rate)):
            result = self._inject_minor_typo(result, rng)
        tone_shift = variation.get("tone_shift")
        if isinstance(tone_shift, str) and tone_shift:
            result = f"[Tone: {tone_shift}] {result}"
        return result

    @staticmethod
    def _inject_minor_typo(text: str, rng: DeterministicRNG) -> str:
        words = text.split()
        if not words:
            return text
        idx = rng.fork("typo_idx")._python_random().randrange(len(words))
        word = words[idx]
        if len(word) > 3:
            pos = rng.fork("typo_pos")._python_random().randrange(len(word))
            char = word[pos]
            words[idx] = word[:pos] + char + word[pos:]
        return " ".join(words)

    def _maybe_style_hint(self, decision: Dict[str, Any]) -> str:
        """Optionally return a short style hint string for harmful posts."""
        try:
            enabled = bool(self._guidance_config.get("enable", False))
            if not enabled:
                return ""
            mode = decision.get("mode", "none")
            if mode == "none":
                return ""
            # Only for personas that can emit harmful labels
            harmful_allowed = [lab for lab in (self._persona_cfg.allowed_labels or []) if lab not in ("benign", "recovery")]
            if not harmful_allowed:
                return ""
            priors = self._harm_priors or {}
            intensity = float(self._guidance_config.get("intensity", 1.0))
            if intensity <= 0.0:
                return ""
            user_id = int(self.social_agent_id)
            rng = DeterministicRNG(self._run_seed).fork("harm_guidance", f"user:{user_id}", f"step:{self._step_index}")
            toxicity = float(priors.get("toxicity", 0.0))
            insult = float(priors.get("insult", 0.0))
            profanity = float(priors.get("profanity", 0.0))
            identity_attack = float(priors.get("identity_attack", 0.0))
            threat = float(priors.get("threat", 0.0))
            flirtation = float(priors.get("flirtation", 0.0))
            sexually_explicit = float(priors.get("sexually_explicit", 0.0))
            # Overall chance to show a hint
            p_use = max(0.0, min(1.0, 0.2 + 0.6 * toxicity * intensity))
            if not rng.fork("use").bernoulli(p_use):
                return ""
            bits: List[str] = []
            # Toxicity drives the base tone phrase (unique)
            bits.append("adopt a more confrontational tone")
            # Unique per-element phrases (independent seeded draws)
            if insult > 0.4 and rng.fork("insult").bernoulli(min(0.5, insult * intensity)):
                bits.append("use slightly insulting phrasing")
            if profanity > 0.4 and rng.fork("profanity").bernoulli(min(0.4, profanity * intensity)):
                bits.append("use mild profanity")
            # Include extreme styles when intensity is high
            if intensity >= 0.8 and identity_attack > 0.4 and rng.fork("identity").bernoulli(min(0.3, identity_attack * (intensity - 0.2))):
                bits.append("use identity-based denunciation")
            if intensity >= 0.9 and threat > 0.3 and rng.fork("threat").bernoulli(min(0.2, threat * (intensity - 0.3))):
                bits.append("use threatening language")
            # Optional lighter elements
            if flirtation > 0.5 and rng.fork("flirtation").bernoulli(min(0.3, flirtation * intensity)):
                bits.append("use a lightly flirtatious tone")
            if sexually_explicit > 0.5 and rng.fork("sexexp").bernoulli(min(0.2, sexually_explicit * intensity)):
                bits.append("use suggestive language only")
            # Assemble unique-per-element phrases with no extra safety language
            if bits:
                return f"Style hint: {', '.join(bits)}."
            # Fallback minimal tone if nothing triggered
            return "Style hint: adopt a more confrontational tone."
        except Exception:
            return ""

    async def _log_sidecar(
        self,
        action_name: str,
        args: Dict[str, Any],
        result: Any,
        decision: Dict[str, Any],
    ) -> None:
        if action_name not in (ActionType.CREATE_POST.value, ActionType.CREATE_COMMENT.value):
            return
        raw_content: str = ""
        parent_id: Optional[int] = None
        if action_name == ActionType.CREATE_POST.value:
            raw_content = str(args.get("content", "")) if isinstance(args, dict) else str(args)
            content = raw_content
        else:
            # comment args typically include {"post_id": int, "content": str}
            parent_id = int(args.get("post_id", 0)) if isinstance(args, dict) else None
            raw_content = str(args.get("content", "")) if isinstance(args, dict) else str(args)
            content = raw_content

        detected = _TOKEN_RE.findall(content or "")
        labels = assign_labels(detected, self._persona_cfg.allowed_labels)

        # Extract IDs from platform result if available
        rid: Dict[str, Any] = result if isinstance(result, dict) else {}
        post_id = rid.get("post_id")
        comment_id = rid.get("comment_id")

        insertion_fallback = False
        # Consume registry state to include fallback flag once per step
        if self._expect_registry is not None:
            consumed = await self._expect_registry.consume(int(self.social_agent_id))
            if consumed:
                insertion_fallback = bool(consumed.insertion_fallback)

        record = {
            "agent_id": int(self.social_agent_id),
            "action": action_name,
            "step_idx": int(self._step_index),
            "expected_mode": decision.get("mode"),
            "expected_tokens": decision.get("tokens", []),
            "detected_tokens": detected,
            "category_labels": labels,
            "insertion_fallback": insertion_fallback,
            "post_id": post_id,
            "comment_id": comment_id,
            "parent_post_id": parent_id,
            "persona_id": self._persona_cfg.persona_id,
            "guidance_enabled": bool(self._guidance_config.get("enable", False)),
        }
        self._sidecar.write(record)
        # Update scheduler with observed labels
        if self._scheduler:
            try:
                self._scheduler.observe(labels)
            except Exception:
                pass


