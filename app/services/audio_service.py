import os
import re
import logging
import requests as _requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import yaml

load_dotenv()
log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "agent.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)["agent"]


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def _clean_for_tts(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = text.replace("...", ", ")
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".?!":
        text += "."
    return text


# ---------------------------------------------------------------------------
# Filler phrases  (played instantly in REST fallback mode)
# ---------------------------------------------------------------------------

_FILLER_PHRASES = {
    "ta": "hmm, seri, oru nimisham please.",
    "en": "hmm, okay, one moment please.",
    "hi": "haan, ek second please.",
    "te": "sare, okka nimisham please.",
    "kn": "sari, ondu nimisha please.",
    "ml": "shari, oru nimisham please.",
}


# ---------------------------------------------------------------------------
# ElevenLabs  (optional fallback if ELEVENLABS_API_KEY is set)
# ---------------------------------------------------------------------------

_EL_BASE     = "https://api.elevenlabs.io/v1"
_EL_MODEL    = "eleven_multilingual_v2"
_EL_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ")
_EL_SETTINGS = {"stability": 0.38, "similarity_boost": 0.78,
                "style": 0.45, "use_speaker_boost": True}

def _el_key()      -> str:  return os.getenv("ELEVENLABS_API_KEY", "")
def _el_headers()  -> dict: return {"xi-api-key": _el_key(),
                                    "Content-Type": "application/json",
                                    "Accept": "audio/mpeg"}
def _el_available() -> bool: return bool(_el_key())


# ---------------------------------------------------------------------------
# AudioService  (public API used by main.py)
# ---------------------------------------------------------------------------

class AudioService:
    OAI_VOICE        = "nova"
    OAI_MODEL        = "gpt-4o-mini-tts"
    OAI_FILLER_MODEL = "tts-1"           # fillers are short — speed matters more here

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.speed  = _CFG["voice"]["rate"]
        self._filler_cache: dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Whisper STT
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, content_type: str = "audio/webm") -> str:
        ext = content_type.split("/")[-1].split(";")[0]
        t = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"audio.{ext}", audio_bytes, content_type),
        )
        return t.text.strip()

    # ------------------------------------------------------------------
    # TTS  — returns (mp3_iter, "audio/mpeg", is_stream)
    # Priority: ElevenLabs → OpenAI
    # ------------------------------------------------------------------

    def speak(self, text: str, lang: str = "ta") -> tuple:
        """
        Returns (data, content_type, is_stream).
        Priority: ElevenLabs → OpenAI TTS
        """
        # ── ElevenLabs ─────────────────────────────────────────────
        if _el_available():
            resp = _requests.post(
                f"{_EL_BASE}/text-to-speech/{_EL_VOICE_ID}/stream",
                json={"text": _clean_for_tts(text),
                      "model_id": _EL_MODEL,
                      "voice_settings": _EL_SETTINGS},
                headers=_el_headers(),
                stream=True, timeout=30,
            )
            resp.raise_for_status()
            return (resp.iter_content(2048), "audio/mpeg", True)

        # ── OpenAI fallback ────────────────────────────────────────
        oai = self.client.audio.speech.create(
            model=self.OAI_MODEL, voice=self.OAI_VOICE,
            input=_clean_for_tts(text), speed=self.speed,
            response_format="mp3",
        )
        return (oai.iter_bytes(2048), "audio/mpeg", True)

    # ------------------------------------------------------------------
    # Filler audio  (always OpenAI — fast, cached, quality fine for 1.5s clip)
    # ------------------------------------------------------------------

    def get_filler(self, lang: str) -> bytes:
        lang = lang if lang in _FILLER_PHRASES else "en"
        if lang not in self._filler_cache:
            resp = self.client.audio.speech.create(
                model=self.OAI_FILLER_MODEL, voice=self.OAI_VOICE,
                input=_FILLER_PHRASES[lang], speed=0.93,
                response_format="mp3",
            )
            self._filler_cache[lang] = resp.content
        return self._filler_cache[lang]

    def prime_fillers(self):
        """Pre-warm filler cache at startup."""
        for lang in _FILLER_PHRASES:
            try:
                self.get_filler(lang)
            except Exception:
                pass


