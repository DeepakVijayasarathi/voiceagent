from __future__ import annotations

import os
import json
import re
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.tenant_service import TenantConfig

load_dotenv()

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "agent.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f)

_CFG = _CONFIG["agent"]


# ---------------------------------------------------------------------------
# Session state — tracks phase + clarifications asked
# ---------------------------------------------------------------------------

class Session:
    def __init__(self):
        self.memory: list        = []
        self.lead: dict          = {
            "name":        None,
            "phone":       None,
            "requirement": None,   # detailed: product + variant + preference
            "budget":      None,
        }
        self.phase: str          = "greeting"   # greeting→explore→recommend→collect→confirm→close
        self.clarify_count: int  = 0            # clarifying Qs asked in explore phase
        self.last_active: float  = time.time()
        self.done: bool          = False


# ---------------------------------------------------------------------------
# Company info helpers — tenant-aware replacements for the old globals.
# These thin wrappers keep callers outside this module simple.
# ---------------------------------------------------------------------------

def get_company_info(tenant: "TenantConfig | None" = None) -> dict:
    """Return company info for the given tenant (falls back to yaml defaults)."""
    if tenant is not None:
        return dict(tenant.company)
    # Fallback: read from yaml (used only during cold-start before registry is ready)
    _yaml_company = _CFG.get("company", {})
    return {
        "name":       _yaml_company.get("name",       "Sales Agent"),
        "tagline":    _yaml_company.get("tagline",    ""),
        "agent_name": _yaml_company.get("agent_name", "Priya"),
        "services":   ", ".join(_yaml_company.get("services", [])),
        "location":   _yaml_company.get("location",   ""),
    }


# ---------------------------------------------------------------------------
# System prompt — phases drive smarter, human-like conversation
# ---------------------------------------------------------------------------

def _get_system_prompt(tenant: "TenantConfig | None" = None) -> str:
    co         = get_company_info(tenant)
    name       = co["name"]
    tagline    = co["tagline"]
    agent_name = co["agent_name"]
    services   = co["services"]

    prompt = f"""You are {agent_name}, a warm, knowledgeable inbound sales agent at {name} — {tagline}.
Products / services sold: {services}.

This is a LIVE PHONE CALL. Every word you write will be spoken aloud by TTS.
Write EXACTLY as you would speak — natural, warm, conversational.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN RULES (never break these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ONE sentence per reply. Max 20 words.
• Ask only ONE question per turn.
• Start every reply with a natural spoken filler: "seri,", "okay,", "hmm,", "aama,", "sure,", "got it,"
• Use commas for breath pauses — NEVER use "..." or markdown.
• Sound like a real helpful human — curious, warm, never robotic.
• NEVER mention you are an AI.
• NEVER ask for info you already have.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION PHASES  ← follow strictly in order
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 — GREETING
  Greet warmly, introduce yourself and {name}, ask how you can help.

PHASE 2 — EXPLORE (most important phase)
  Goal: Deeply understand what the customer REALLY needs before collecting any data.
  • First ask WHAT product/service they want.
  • Then ask 1–2 smart clarifying questions based on the product type:
      – Phone   → Android or iPhone? Main use (camera, gaming, battery)?
      – TV      → Room size? Smart TV or basic? Preferred brand?
      – Laptop  → Work or gaming? Which software do they use?
      – Washing machine → Family size? Front load or top load?
      – AC      → Room size in sq.ft? Inverter preferred?
      – General → What features matter most to them?
  • Listen to their answers and ask naturally — like a friend who knows the store well.
  • After 2 clarifying questions MAX, move to PHASE 3.
  • If customer seems to already know exactly what they want, skip clarifying and recommend directly.

PHASE 3 — RECOMMEND
  • Using the [RETRIEVED CONTEXT] (if available), suggest 1–2 specific products.
  • Include the product name and price naturally in speech.
  • Example: "seri, Samsung Galaxy S24 nalla match aagum, ₹79,999 la irukku, camera romba nalla irukku."
  • If no context available, acknowledge their need and say the team will help find the best option.
  • After recommending, ask if that sounds good or if they want to explore other options.
  • Once customer shows interest → move to PHASE 4.

PHASE 4 — COLLECT LEAD INFO (after requirement is clear)
  Collect in this order — ask one per turn:
  a) Name   → "unga peyar enna sollunga?" / "may I have your name please?"
  b) Phone  → "unga phone number sollunga please." / "and your phone number?"
  c) Budget → only if NOT already mentioned during explore. Skip if already known.

PHASE 5 — CONFIRM
  Read back: name, requirement (with product name if known), phone, budget.
  Ask "correct aa?" / "is that right?"

PHASE 6 — CLOSE
  Thank warmly. Say team will call shortly. Set done: true.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HANDLING SPECIFIC SITUATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Customer asks price   → give exact price from [RETRIEVED CONTEXT] if available.
• Customer says "too expensive" → acknowledge, suggest a lower-priced alternative from context.
• Customer is vague ("just checking") → gently ask what brought them to the store today.
• Customer says "I don't know" → suggest the most popular product in that category.
• Customer interrupts / changes topic → acknowledge and answer, then gently guide back.
• Customer gives info you didn't ask for → acknowledge it, save it in lead, continue naturally.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE — DEFAULT: TAMIL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start in Tamil. Switch ONLY if customer clearly speaks another language.
Match the customer's style: formal if they're formal, friendly if they're casual.

TAMIL (Tanglish — warm, professional):
  Fillers: "seri,", "okay,", "aama,", "nandri,", "hmm,", "achaa,"
  Formal: "neenga / unga" — NEVER "da / di / machan"
  Intro:    "vanakkam, naan {agent_name} pesuraen {name} la irundhu, enna seiya?"
  Explore:  "seri, enna maatiri product theduraanga?"
  Clarify:  "okay, [clarifying question based on product]?"
  Recommend:"seri, [product] romba nalla irukku, [price] la — [one key feature]."
  Collect:  "aama, unga peyar enna sollunga?" / "nandri, phone number?"
  Budget:   "seri, unga budget range enna?" (skip if already shared)
  Confirm:  "okay, confirm — [name], [requirement], [phone], [budget], correct aa?"
  Close:    "romba nandri! Namma team ungaluku call pannuvanga!"

ENGLISH (warm, professional):
  Fillers: "okay,", "right,", "sure,", "got it,", "of course,"
  Intro:    "hello, thanks for calling {name}, I'm {agent_name}, how can I help?"
  Clarify:  "right, [clarifying question based on product]?"
  Recommend:"sure, the [product] would be great for you, it's ₹[price] and [one feature]."
  Confirm:  "let me confirm — [name], [need], [phone], [budget] — is that correct?"
  Close:    "thank you so much, our team will call you shortly!"

HINDI (Hinglish):    Fillers: "haan,", "bilkul,", "achha,", "theek hai,"
TELUGU (Tenglish):   Fillers: "sare,", "dhanyavaadaalu,", "okay,"
KANNADA (Kanglish):  Fillers: "sari,", "houdu,", "okay,", "thumba thanks,"
MALAYALAM (Manglish): Fillers: "ശരി (shari),", "okay,", "nandi,", "manasilaayi,"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHONE VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
India: 10 digits, starts with 6–9. If invalid → null, ask again politely once.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — ONLY valid JSON, no other text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "reply":   "<one warm spoken sentence, filler start, max 20 words>",
  "lang":    "<ta|en|hi|te|kn|ml — match customer's language>",
  "emotion": "<interested|curious|hesitant|budget_concern|satisfied|neutral>",
  "phase":   "<greeting|explore|recommend|collect|confirm|close>",
  "lead": {{
    "name":        "<full name or null>",
    "phone":       "<10-digit India mobile or null>",
    "requirement": "<detailed requirement: product + variant/size + preference, or null>",
    "budget":      "<stated budget or range or null>"
  }},
  "done": <true only after PHASE 6 close is complete, else false>
}}"""

    # NOTE: _knowledge_base presence check is done at the call site (tenant.knowledge_base.is_loaded)
    # so the prompt block below is injected only when a KB is loaded.
    return prompt


def _get_system_prompt_with_kb(tenant: "TenantConfig | None" = None) -> str:
    """Full system prompt, appending the KB instruction block when a KB is loaded."""
    prompt = _get_system_prompt(tenant)
    kb_loaded = (tenant is not None and tenant.knowledge_base.is_loaded) or \
                (tenant is None and False)
    if kb_loaded:
        prompt += (
            "\n\n━━━ KNOWLEDGE BASE LOADED ━━━\n"
            "Relevant product/price/policy sections from the company document will appear in "
            "[RETRIEVED CONTEXT] before each user message.\n"
            "• ALWAYS quote exact prices and product names from that context.\n"
            "• If a product matches the customer's requirement, recommend it confidently.\n"
            "• If price is asked and context has it, give it directly — don't say 'I\'ll check\'.\n"
            "• If context has no relevant info for a question, say 'let me check and confirm shortly'."
        )
    return prompt


# ---------------------------------------------------------------------------
# Agent service
# ---------------------------------------------------------------------------

class AgentService:

    MODEL            = "gpt-4o"
    TEMPERATURE      = 0.65
    MAX_TOKENS       = 280
    MEMORY_LIMIT     = 14        # remember last 14 turns (7 full exchanges)
    SESSION_TIMEOUT  = 600       # 10 minutes

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # sessions[tenant_id][session_id] = Session
        self._sessions: dict[str, dict[str, Session]] = {}

        # Default LLM settings from yaml (overridden per-call by tenant config)
        llm = _CFG.get("llm", {})
        self._default_model       = llm.get("model",       self.MODEL)
        self._default_temperature = llm.get("temperature", self.TEMPERATURE)
        self._default_max_tokens  = llm.get("max_tokens",  self.MAX_TOKENS)
        conv = _CFG.get("conversation", {})
        self.memory_limit    = conv.get("memory_limit",        self.MEMORY_LIMIT)
        self.session_timeout = conv.get("session_timeout_sec", self.SESSION_TIMEOUT)

    def _llm_settings(self, tenant: "TenantConfig | None") -> tuple[str, float, int]:
        """Return (model, temperature, max_tokens) for the given tenant."""
        if tenant is None:
            return self._default_model, self._default_temperature, self._default_max_tokens
        llm = tenant.llm
        return (
            llm.get("model",       self._default_model),
            llm.get("temperature", self._default_temperature),
            llm.get("max_tokens",  self._default_max_tokens),
        )

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _get_session(self, session_id: str, tenant_id: str = "default") -> Session:
        bucket = self._sessions.setdefault(tenant_id, {})
        now     = time.time()
        expired = [sid for sid, s in bucket.items()
                   if now - s.last_active > self.session_timeout]
        for sid in expired:
            del bucket[sid]
        if session_id not in bucket:
            bucket[session_id] = Session()
        session = bucket[session_id]
        session.last_active = now
        return session

    def clear_session(self, session_id: str, tenant_id: str = "default"):
        self._sessions.get(tenant_id, {}).pop(session_id, None)

    def get_lead(self, session_id: str, tenant_id: str = "default") -> dict | None:
        s = self._sessions.get(tenant_id, {}).get(session_id)
        return dict(s.lead) if s else None

    # ------------------------------------------------------------------
    # Phone validation (India)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_phone(raw) -> str | None:
        if not raw:
            return None
        digits = re.sub(r"\D", "", str(raw))
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) == 10 and digits[0] in "6789":
            return digits
        return None

    # ------------------------------------------------------------------
    # Build messages list with RAG context injected
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        session: Session,
        message: str,
        is_trigger: bool,
        detected_lang: str | None = None,
        tenant: "TenantConfig | None" = None,
    ) -> list:
        messages: list = [
            {"role": "system", "content": _get_system_prompt_with_kb(tenant)},
            *session.memory[-self.memory_limit:],
        ]

        if is_trigger:
            messages.append({"role": "user", "content": "Begin the call now."})
            return messages

        # RAG: retrieve relevant PDF sections for the user's message
        kb = tenant.knowledge_base if tenant is not None else None
        if kb and kb.is_loaded:
            # Enrich query with known requirement for higher-precision retrieval
            query = message
            if session.lead.get("requirement"):
                query = f"{session.lead['requirement']} {message}"
            context = kb.get_context(query)
            if context:
                # Insert just before the last user message
                messages.insert(-1, {
                    "role": "system",
                    "content": (
                        "[RETRIEVED CONTEXT — quote prices and product names from here]\n\n"
                        + context
                    ),
                })

        # ── Detected language — inject IMMEDIATELY so reply switches this turn ──
        if detected_lang:
            _LANG_NAMES = {
                "ta": "Tamil/Tanglish",
                "en": "English",
                "hi": "Hindi/Hinglish",
                "te": "Telugu/Tenglish",
                "kn": "Kannada/Kanglish",
                "ml": "Malayalam/Manglish",
                "mr": "Marathi",
                "bn": "Bengali",
            }
            lang_name = _LANG_NAMES.get(detected_lang, detected_lang.upper())
            messages.insert(-1, {
                "role": "system",
                "content": (
                    f"[LANGUAGE DETECTED: {lang_name} ({detected_lang})]\n"
                    f"The customer just spoke in {lang_name}. "
                    f"You MUST reply in {lang_name} starting from THIS turn. "
                    f"Do not wait for the next turn. Set \"lang\": \"{detected_lang}\" in your JSON."
                ),
            })

        # ── Already-collected lead state — never re-ask ───────────────────────
        collected = {k: v for k, v in session.lead.items() if v}
        if collected:
            messages.insert(1, {
                "role": "system",
                "content": (
                    "[ALREADY COLLECTED — do NOT ask for these again]\n"
                    + "\n".join(f"• {k}: {v}" for k, v in collected.items())
                    + f"\n• current phase: {session.phase}"
                ),
            })

        return messages

    # ------------------------------------------------------------------
    # Main handler
    # ------------------------------------------------------------------

    def handle(
        self,
        session_id: str,
        message: str,
        detected_lang: str | None = None,
        tenant: "TenantConfig | None" = None,
    ) -> dict:
        tenant_id  = tenant.tenant_id if tenant is not None else "default"
        session    = self._get_session(session_id, tenant_id)
        is_trigger = message.startswith("__")

        if session.done:
            return {
                "reply":   "We already have your details — our team will call you shortly!",
                "lang":    "en",
                "emotion": "satisfied",
                "phase":   "close",
                "lead":    dict(session.lead),
                "done":    True,
            }

        if not is_trigger:
            session.memory.append({"role": "user", "content": message})

        messages = self._build_messages(session, message, is_trigger, detected_lang, tenant)
        model, temperature, max_tokens = self._llm_settings(tenant)

        try:
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            # Robust JSON parse — try direct parse first, then regex extraction
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                match = re.search(r'\{.*\}', raw or "", re.DOTALL)
                data = json.loads(match.group(0)) if match else {}
        except (json.JSONDecodeError, TypeError):
            data = {
                "reply":   "hmm, seri, oru nimisham please.",
                "lang":    "ta",
                "emotion": "neutral",
                "phase":   session.phase,
                "lead":    {},
                "done":    False,
            }
        except Exception:
            data = {
                "reply":   "sorry, oru nimisham technical issue — please bear with me.",
                "lang":    "en",
                "emotion": "neutral",
                "phase":   session.phase,
                "lead":    {},
                "done":    False,
            }

        # ── Accumulate lead fields across turns ─────────────────────────
        incoming_lead = data.get("lead", {})
        for field in ("name", "requirement", "budget"):
            val = incoming_lead.get(field)
            if val and str(val).lower() not in ("null", "none", ""):
                session.lead[field] = val

        validated = self._validate_phone(incoming_lead.get("phone"))
        if validated:
            session.lead["phone"] = validated

        data["lead"] = dict(session.lead)

        # ── Phase tracking ───────────────────────────────────────────────
        new_phase = data.get("phase", session.phase)
        if new_phase in ("greeting", "explore", "recommend", "collect", "confirm", "close"):
            if new_phase == "explore" and session.phase == "explore":
                session.clarify_count += 1
            session.phase = new_phase

        # ── done gate: all 4 fields must be collected ────────────────────
        all_collected = all(session.lead.get(f) for f in ("name", "phone", "requirement", "budget"))
        done = all_collected and bool(data.get("done", False))
        data["done"] = done
        session.done = done

        # ── Store assistant reply in memory ──────────────────────────────
        session.memory.append({"role": "assistant", "content": data.get("reply", "")})

        return data
