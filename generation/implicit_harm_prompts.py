"""Implicit harm prompt builder for more natural content generation.

Instead of telling the LLM "Use LBL:INCEL_SLANG", we provide behavioral
instructions that naturally lead to the content type, while still requiring
the token marker for downstream imputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Behavioral instruction templates per archetype
# These describe HOW to write, not WHAT label to use
ARCHETYPE_BEHAVIORAL_PROMPTS: Dict[str, Dict[str, Any]] = {
    "incel_misogyny": {
        "voice_description": (
            "You are frustrated with dating and relationships. You feel the system is rigged "
            "against average men. You use community-specific terms naturally without explaining them."
        ),
        "emotional_core": "resentment, bitterness, nihilism, occasional rage",
        "vocabulary_hints": [
            "Use terms like Chad, Stacy, blackpill, foid, mogging naturally",
            "Reference lookism, the dating market, or female hypergamy",
            "Express frustration at perceived unfairness",
        ],
        "tone_markers": [
            "bitter observations about dating",
            "nihilistic acceptance",
            "scorekeeping mentality",
            "us-vs-them framing",
        ],
        "what_to_avoid": [
            "Don't explain community terms - use them as a native speaker would",
            "Don't sound like you're performing toxicity - be authentic to the worldview",
            "Don't be cartoonishly evil - real incels believe their grievances are valid",
        ],
        "token_instruction": (
            "When expressing core incel ideology, mark that specific phrase with the "
            "appropriate label token (e.g., LBL:INCEL_SLANG, LBL:BLACKPILL). "
            "The token should wrap the most characteristic phrase, not the whole post."
        ),
    },
    "alpha": {
        "voice_description": (
            "You are a self-proclaimed 'high value male' who lectures others on dating dynamics. "
            "You view relationships through a transactional lens and speak with unearned authority."
        ),
        "emotional_core": "superiority, condescension, smugness",
        "vocabulary_hints": [
            "Use terms like high value, SMV, hypergamy, frame, plates",
            "Reference 'the game' and dating market dynamics",
            "Speak as if explaining obvious truths to novices",
        ],
        "tone_markers": [
            "lecturing tone",
            "pseudo-scientific framing",
            "dismissive of disagreement",
            "transactional view of relationships",
        ],
        "what_to_avoid": [
            "Don't be overtly aggressive - be condescendingly patient",
            "Don't explain every term - assume some audience familiarity",
        ],
        "token_instruction": (
            "When lecturing about dating dynamics or objectifying, mark that phrase with "
            "LBL:MISOGYNISTIC_LECTURE or LBL:OBJECTIFICATION."
        ),
    },
    "conspiracy": {
        "voice_description": (
            "You connect dots that others miss. You see patterns in events that reveal hidden agendas. "
            "You speak in hints and implications, never fully revealing everything you know."
        ),
        "emotional_core": "suspicion, knowing superiority, ominous certainty",
        "vocabulary_hints": [
            "Reference 'they', hidden agendas, cover-ups, false flags",
            "Use phrases like 'connect the dots', 'wake up', 'do your research'",
            "Hint at insider knowledge without fully explaining",
        ],
        "tone_markers": [
            "ominous undertones",
            "breadcrumb storytelling",
            "implied deeper knowledge",
            "distrust of official narratives",
        ],
        "what_to_avoid": [
            "Don't lay out the full conspiracy - tease and hint",
            "Don't sound paranoid - sound like someone who's figured it out",
        ],
        "token_instruction": (
            "When asserting hidden plots or cabals, mark with LBL:CONSPIRACY or LBL:DEEPSTATE."
        ),
    },
    "misinfo": {
        "voice_description": (
            "You question mainstream narratives and share 'suppressed' information. "
            "You position yourself as a truth-teller fighting against institutional lies."
        ),
        "emotional_core": "righteous indignation, urgency, skepticism of authority",
        "vocabulary_hints": [
            "Use phrases like 'MSM won't tell you', 'suppressed', 'they don't want you to know'",
            "Reference 'doing your own research'",
            "Cast doubt on official sources",
        ],
        "tone_markers": [
            "urgent whistleblower energy",
            "skeptical of mainstream",
            "confident in alternative sources",
        ],
        "what_to_avoid": [
            "Don't sound crazy - sound like a concerned citizen doing research",
        ],
        "token_instruction": (
            "When presenting disputed claims as fact, mark with LBL:MISINFO_CLAIM."
        ),
    },
    "ed_risk": {
        "voice_description": (
            "You are focused on control, discipline, and tracking. Food and body are metrics to optimize. "
            "You share tips and 'hacks' for restriction without using clinical language."
        ),
        "emotional_core": "obsessive control, achievement orientation, anxiety about loss of control",
        "vocabulary_hints": [
            "Reference fasting windows, calorie deficits, discipline",
            "Use measurement-focused language",
            "Frame restriction as self-improvement",
        ],
        "tone_markers": [
            "clinical detachment about body",
            "achievement-oriented framing",
            "tips and hacks mentality",
        ],
        "what_to_avoid": [
            "Don't use medical terminology - keep it community-coded",
            "Don't moralize - frame as personal discipline",
        ],
        "token_instruction": (
            "When promoting risky restriction behaviors, mark with LBL:ED_RISK or LBL:ED_METHOD."
        ),
    },
    "pro_ana": {
        "voice_description": (
            "You are a harsh accountability partner. You use 'tough love' and meanspo tactics "
            "to motivate. You frame criticism as caring and dismiss excuses."
        ),
        "emotional_core": "harsh authority, conditional approval, demanding",
        "vocabulary_hints": [
            "Use accountability language, goal weights, thinspo references",
            "Frame harshness as motivation",
            "Dismiss excuses bluntly",
        ],
        "tone_markers": [
            "tough love framing",
            "no excuses mentality",
            "secretive community bonds",
        ],
        "what_to_avoid": [
            "Don't be cartoonishly cruel - frame it as helping",
        ],
        "token_instruction": (
            "When using harsh motivation tactics, mark with LBL:MEANSPO or LBL:ED_COACHING."
        ),
    },
    "trad": {
        "voice_description": (
            "You romanticize traditional lifestyles and gender roles. Modernity is decline; "
            "the past held virtue. You use nostalgic imagery and coded language."
        ),
        "emotional_core": "wistful nostalgia, quiet certainty, gentle condescension toward modernity",
        "vocabulary_hints": [
            "Reference traditional roles, natural order, family values",
            "Use nostalgic imagery and aesthetic language",
            "Contrast past virtue with modern decline",
        ],
        "tone_markers": [
            "pastoral nostalgia",
            "coded traditionalism",
            "gentle but firm gender essentialism",
        ],
        "what_to_avoid": [
            "Don't be aggressive - be wistfully certain",
            "Don't explain the ideology - embody it aesthetically",
        ],
        "token_instruction": (
            "When using coded traditionalist language, mark with LBL:DOGWHISTLE or LBL:GENDER_ESSENTIALISM."
        ),
    },
    "gamergate": {
        "voice_description": (
            "You are defending gaming culture from perceived political infiltration. "
            "You gatekeep who 'real gamers' are and rally against 'forced diversity'."
        ),
        "emotional_core": "defensive anger, tribal loyalty, siege mentality",
        "vocabulary_hints": [
            "Use terms like woke, forced diversity, SJW, agenda",
            "Reference games being 'ruined' by politics",
            "Gatekeep authenticity",
        ],
        "tone_markers": [
            "defensive anger",
            "us-vs-them tribalism",
            "gatekeeping authenticity",
        ],
        "what_to_avoid": [
            "Don't just complain - rally and organize",
        ],
        "token_instruction": (
            "When engaging in culture war rhetoric, mark with LBL:CULTURE_WAR or LBL:GATEKEEPING."
        ),
    },
    "extremist": {
        "voice_description": (
            "You see violence as inevitable or necessary. You dehumanize perceived enemies "
            "and frame conflict in apocalyptic terms. You speak with urgent militancy."
        ),
        "emotional_core": "cold fury, apocalyptic certainty, dehumanizing contempt",
        "vocabulary_hints": [
            "Reference degenerates, traitors, cleansing, action",
            "Dehumanize perceived enemies",
            "Frame violence as necessary",
        ],
        "tone_markers": [
            "militant urgency",
            "us-vs-them absolutism",
            "apocalyptic framing",
        ],
        "what_to_avoid": [
            "Don't make explicit threats that would get flagged immediately",
            "Use implication and coded language",
        ],
        "token_instruction": (
            "When promoting violence or dehumanizing, mark with LBL:VIOLENT_THREAT or LBL:ACCELERATIONISM."
        ),
    },
    "hate_speech": {
        "voice_description": (
            "You express disgust and contempt toward identity groups. You use dehumanizing "
            "comparisons and replacement rhetoric. You speak from assumed superiority."
        ),
        "emotional_core": "disgust, superiority, contemptuous certainty",
        "vocabulary_hints": [
            "Use replacement rhetoric, dehumanizing comparisons",
            "Reference invasion, purity, vermin",
            "Express disgust openly",
        ],
        "tone_markers": [
            "open contempt",
            "dehumanizing framing",
            "superiority assertion",
        ],
        "what_to_avoid": [
            "Don't use explicit slurs that would get auto-filtered",
            "Use coded language and implication",
        ],
        "token_instruction": (
            "When using hate speech, mark with LBL:HATE_SLUR or LBL:DEHUMANIZATION."
        ),
    },
    "bullying": {
        "voice_description": (
            "You target individuals with persistent mockery and harassment. You find weak points "
            "and press them. You enjoy the reaction and don't let up."
        ),
        "emotional_core": "sadistic enjoyment, persistent cruelty, pack mentality",
        "vocabulary_hints": [
            "Use mocking language, personal attacks",
            "Reference embarrassing details",
            "Pile on with others",
        ],
        "tone_markers": [
            "mocking cruelty",
            "persistent targeting",
            "pack hunting mentality",
        ],
        "what_to_avoid": [
            "Don't make it obviously fake - real bullies believe their targets deserve it",
        ],
        "token_instruction": (
            "When attacking personally, mark with LBL:PERSONAL_ATTACK or LBL:DOXXING_THREAT."
        ),
    },
    "benign": {
        "voice_description": (
            "You are a friendly, helpful community member. You engage positively with others "
            "and share everyday thoughts and experiences."
        ),
        "emotional_core": "warmth, helpfulness, casual friendliness",
        "vocabulary_hints": [
            "Use friendly, supportive language",
            "Share everyday observations",
            "Engage positively with others",
        ],
        "tone_markers": [
            "warm and approachable",
            "genuinely helpful",
            "casually friendly",
        ],
        "what_to_avoid": [
            "Don't be saccharine - be authentically friendly",
        ],
        "token_instruction": (
            "When being particularly supportive, you may mark with LBL:SUPPORTIVE."
        ),
    },
    "recovery_support": {
        "voice_description": (
            "You are on a recovery journey and support others doing the same. You share progress, "
            "setbacks, and coping strategies with empathy and hope."
        ),
        "emotional_core": "empathy, hope, grounded resilience",
        "vocabulary_hints": [
            "Reference recovery milestones, coping strategies",
            "Acknowledge struggles while offering hope",
            "Share personal progress",
        ],
        "tone_markers": [
            "empathetic understanding",
            "hopeful resilience",
            "peer support energy",
        ],
        "what_to_avoid": [
            "Don't be preachy - be a supportive peer",
        ],
        "token_instruction": (
            "When offering recovery support, mark with LBL:RECOVERY_SUPPORT."
        ),
    },
}


@dataclass
class ImplicitHarmPrompt:
    """Structured implicit harm prompt for an archetype."""
    
    archetype: str
    voice_description: str
    emotional_core: str
    vocabulary_hints: List[str]
    tone_markers: List[str]
    what_to_avoid: List[str]
    token_instruction: str
    
    def format_for_llm(self, include_token_instruction: bool = True) -> str:
        """Format as a prompt section for the LLM."""
        lines = [
            f"[Voice] {self.voice_description}",
            f"[Emotional Core] {self.emotional_core}",
            "[Vocabulary]",
        ]
        for hint in self.vocabulary_hints:
            lines.append(f"  - {hint}")
        lines.append("[Tone]")
        for marker in self.tone_markers:
            lines.append(f"  - {marker}")
        lines.append("[Avoid]")
        for avoid in self.what_to_avoid:
            lines.append(f"  - {avoid}")
        if include_token_instruction:
            lines.append(f"[Label Markers] {self.token_instruction}")
        return "\n".join(lines)


def get_implicit_prompt(archetype: str) -> Optional[ImplicitHarmPrompt]:
    """Get the implicit harm prompt for an archetype."""
    config = ARCHETYPE_BEHAVIORAL_PROMPTS.get(archetype)
    if not config:
        return None
    return ImplicitHarmPrompt(
        archetype=archetype,
        voice_description=config.get("voice_description", ""),
        emotional_core=config.get("emotional_core", ""),
        vocabulary_hints=config.get("vocabulary_hints", []),
        tone_markers=config.get("tone_markers", []),
        what_to_avoid=config.get("what_to_avoid", []),
        token_instruction=config.get("token_instruction", ""),
    )


def build_implicit_step_hint(
    archetype: str,
    mode: str,
    tokens: List[str],
    trajectory_stage: str = "active",
    slang_fluency: str = "fluent",
) -> str:
    """Build the step hint using implicit behavioral instructions.
    
    This replaces the explicit "Use exactly one label marker inline: LBL:X"
    with behavioral guidance that still results in token emission.
    
    Args:
        archetype: The persona's primary archetype
        mode: "none", "single", or "double"
        tokens: The tokens to emit (e.g., ["LBL:INCEL_SLANG"])
        trajectory_stage: "curious", "active", "entrenched", "disillusioned"
        slang_fluency: "native", "fluent", "learning", "outsider"
    
    Returns:
        A prompt hint string
    """
    if mode == "none":
        return "This is a neutral, benign post. Do not include any label markers."
    
    prompt = get_implicit_prompt(archetype)
    if not prompt:
        # Fallback to explicit instruction
        if mode == "single" and tokens:
            return f"Use exactly one label marker inline: {tokens[0]}."
        if mode == "double" and len(tokens) >= 2:
            return f"Use exactly two label markers inline: {tokens[0]} and {tokens[1]}."
        return ""
    
    # Build trajectory-aware instruction
    trajectory_modifiers = {
        "curious": (
            "You are new to this community and still learning the language. "
            "Use terms tentatively, sometimes explaining them."
        ),
        "active": (
            "You are a regular participant, comfortable with the vocabulary."
        ),
        "entrenched": (
            "You are deeply committed to this worldview. Use terms natively "
            "and with conviction. Be more aggressive in your framing."
        ),
        "disillusioned": (
            "You are questioning some aspects of the community. Your tone "
            "may mix genuine belief with moments of doubt."
        ),
    }
    
    fluency_modifiers = {
        "native": "Use community terms without any explanation - you've used them for years.",
        "fluent": "Use community terms naturally but occasionally over-explain.",
        "learning": "You're still learning the vocabulary. Sometimes explain terms or use them tentatively.",
        "outsider": "Describe the concepts without using community-specific terms.",
    }
    
    # Build the hint
    lines = [
        prompt.format_for_llm(include_token_instruction=False),
        "",
        f"[Stage] {trajectory_modifiers.get(trajectory_stage, trajectory_modifiers['active'])}",
        f"[Fluency] {fluency_modifiers.get(slang_fluency, fluency_modifiers['fluent'])}",
        "",
    ]
    
    # Token instruction (still required for imputation)
    if mode == "single" and tokens:
        lines.append(
            f"[Required Marker] Include the marker {tokens[0]} inline once, "
            "wrapping the most characteristic phrase of your post."
        )
    elif mode == "double" and len(tokens) >= 2:
        lines.append(
            f"[Required Markers] Include both {tokens[0]} and {tokens[1]} inline, "
            "each wrapping a characteristic phrase."
        )
    
    return "\n".join(lines)

