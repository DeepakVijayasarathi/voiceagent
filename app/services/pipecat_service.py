"""
Pipecat 1.0.0 real-time voice pipeline — advanced multilingual.

Improvements vs previous version:
  • Whisper verbose_json  → captures detected language on every transcription
  • Unicode script scan   → instant Tamil/Hindi/Telugu detection without LLM
  • Detected lang injected into agent SAME turn (not next turn)
  • Per-language TTS instructions with precise phonetic guidance
  • Dynamic TTS settings update on language switch (Pipecat set_model/set_voice)
"""

import asyncio
import os
import re
import struct

from loguru import logger
from fastapi import WebSocket

from pipecat.serializers.base_serializer import FrameSerializer


# ---------------------------------------------------------------------------
# Language utilities
# ---------------------------------------------------------------------------

# Whisper returns full language names — map to our 2-letter codes
_WHISPER_LANG_MAP: dict[str, str] = {
    "tamil":     "ta",
    "english":   "en",
    "hindi":     "hi",
    "telugu":    "te",
    "kannada":   "kn",
    "malayalam": "ml",
    "urdu":      "ur",
    "bengali":   "bn",
    "marathi":   "mr",
}

# Unicode ranges for instant script detection (no API call needed)
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0B80, 0x0BFF, "ta"),   # Tamil script
    (0x0900, 0x097F, "hi"),   # Devanagari  (Hindi / Marathi)
    (0x0C00, 0x0C7F, "te"),   # Telugu script
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0C80, 0x0CFF, "kn"),   # Kannada
]


def _detect_script(text: str) -> str | None:
    """
    Instantly detect non-Latin script from Unicode codepoints.
    Returns 2-letter lang code or None (= Latin, could be English or Tanglish).
    """
    for char in text:
        cp = ord(char)
        for lo, hi, code in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return code
    return None


def _whisper_lang_to_code(whisper_lang: str) -> str | None:
    """Convert Whisper's full language name to our 2-letter code."""
    return _WHISPER_LANG_MAP.get(whisper_lang.lower().strip()) if whisper_lang else None


# ---------------------------------------------------------------------------
# Per-language TTS instructions (precise phonetics for each)
# ---------------------------------------------------------------------------

_TTS_INSTRUCTIONS: dict[str, str] = {
    "ta": (
        "You are Priya, a warm Tamil-speaking Indian call centre agent from Chennai. "
        "Pronunciation rules:\n"
        "• Tamil retroflex consonants ட ண ள ழ — tongue curls back to the palate.\n"
        "• The unique Tamil 'zh' (ழ) sound — no equivalent in English, like a retroflex 'l'.\n"
        "• Nasal ம and ந — fully nasal, not nasalized vowels.\n"
        "• Tanglish words (Tamil grammar, English vocabulary): keep English words in natural "
        "Indian-English accent but use Tamil sentence rhythm and intonation.\n"
        "• Formal pronouns 'neenga/unga' — warm, respectful.\n"
        "• Sentence-final question: voice rises gently on the last syllable.\n"
        "Speak at a warm, conversational pace — like a real Madurai call centre agent. "
        "Natural breath after every comma."
    ),
    "en": (
        "You are Priya, a warm Indian-English call centre agent. "
        "Speak in clear, professional Indian English — slight South Indian accent is natural. "
        "Aspirated stops (p, t, k) are lightly aspirated. "
        "Vowels are pure, not diphthongized. "
        "Voice rises gently on questions, falls decisively on statements. "
        "Warm, helpful, never robotic. Natural pace — not rushed."
    ),
    "hi": (
        "You are Priya, a warm Hindi-speaking Indian call centre agent. "
        "Pronunciation rules:\n"
        "• Aspirated consonants ख घ छ झ ठ ढ — strong puff of air.\n"
        "• Retroflex consonants ट ड ण — tongue tip to the hard palate.\n"
        "• Nasal anusvara (ं) — fully nasal, context-dependent place.\n"
        "• Hinglish: Hindi grammar with English words — English words keep Indian-English accent.\n"
        "Speak in warm Delhi/Mumbai call centre style. Natural rhythm with pauses after commas."
    ),
    "te": (
        "You are Priya, a warm Telugu-speaking Indian call centre agent from Hyderabad. "
        "Pronunciation rules:\n"
        "• Telugu retroflex ట డ ణ — tongue curves back.\n"
        "• Short vs long vowels are phonemically distinct — honour the length.\n"
        "• Tenglish: Telugu sentence structure with English words — fluid and natural.\n"
        "Warm, professional Hyderabad call centre tone. Gentle rising intonation on questions."
    ),
    # Fallback for any unrecognised language
    "_default": (
        "You are Priya, a warm and knowledgeable Indian call centre agent. "
        "Speak naturally — warm, conversational, never robotic. "
        "Use natural rhythm with gentle pauses after commas and a rising tone for questions. "
        "Honour the phonetics of whatever Indian language you are speaking."
    ),
}


def _get_tts_instructions(lang: str) -> str:
    return _TTS_INSTRUCTIONS.get(lang, _TTS_INSTRUCTIONS["_default"])


# ---------------------------------------------------------------------------
# Transcript quality filter
# ---------------------------------------------------------------------------

# Single filler words that Whisper hallucinates on silence / noise
_NOISE_ONLY: set[str] = {
    "you", "the", "a", "an", "i", "yes", "no", "okay", "ok", "hmm", "hm",
    "uh", "um", "ah", "oh", "what", "does", "do", "is", "are", "was",
    "thanks", "thank", "bye", "hi", "hello", "hey",
    "um-hum", "uh-huh", "mm", "mm-hmm", "right",
}


def _is_valid_transcript(text: str) -> bool:
    """
    Return False for transcripts that are almost certainly noise / false triggers:
      • Fewer than 3 words  AND fewer than 10 characters
      • Single word that is a known Whisper hallucination on silence
    """
    text   = text.strip()
    words  = text.split()
    n_words = len(words)

    # Single word — suppress unless it's a clear product/command keyword
    if n_words == 1:
        word = words[0].lower().rstrip(".,!?")
        # Always suppress known noise/filler words
        if word in _NOISE_ONLY:
            logger.info("Transcript suppressed (noise word): {!r}", text)
            return False
        # Suppress any other single-word utterance under 14 chars
        # (avoids "DigiTechzo", "Samsung" etc. as isolated triggers)
        if len(text) <= 14:
            logger.info("Transcript suppressed (single word): {!r}", text)
            return False

    # Two-word utterances under 12 chars (e.g. "okay yes", "hi there")
    if n_words == 2 and len(text) < 12:
        logger.info("Transcript suppressed (two-word noise): {!r}", text)
        return False

    return True


# ---------------------------------------------------------------------------
# TTS text cleaner
# ---------------------------------------------------------------------------

def _clean_for_tts(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = text.replace('...', ', ')
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[-1] not in '.?!':
        text += '.'
    return text


# ---------------------------------------------------------------------------
# Helper: WAV header
# ---------------------------------------------------------------------------

def _wav_bytes(pcm: bytes, sample_rate: int, num_channels: int) -> bytes:
    bits  = 16
    block = num_channels * bits // 8
    rate  = sample_rate * block
    hdr   = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE",
        b"fmt ", 16, 1, num_channels, sample_rate, rate, block, bits,
        b"data", len(pcm),
    )
    return hdr + pcm


# ---------------------------------------------------------------------------
# Custom serializer — raw PCM ↔ WAV
# ---------------------------------------------------------------------------

class RawPCMFrameSerializer(FrameSerializer):
    _IN_RATE = 16_000
    _IN_CH   = 1

    async def serialize(self, frame) -> bytes | None:
        from pipecat.frames.frames import OutputAudioRawFrame
        if isinstance(frame, OutputAudioRawFrame):
            audio = frame.audio
            if not audio:
                return None
            if audio[:4] == b"RIFF":
                return audio
            return _wav_bytes(audio, frame.sample_rate, frame.num_channels)
        return None

    async def deserialize(self, data: bytes | str):
        from pipecat.frames.frames import InputAudioRawFrame
        if isinstance(data, (bytes, bytearray)) and data:
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=self._IN_RATE,
                num_channels=self._IN_CH,
            )
        return None


# ---------------------------------------------------------------------------
# Multilingual STT — whisper-1 with verbose_json for language detection
# ---------------------------------------------------------------------------

def _make_multilingual_stt(api_key: str):
    """
    Whisper-1 STT with:
    • No language= → auto-detects Tamil/EN/HI/TE
    • verbose_json → exposes result.language (e.g. "tamil")
    • Domain hint from knowledge base → better product-name transcription
    """
    from pipecat.services.openai.stt import OpenAISTTService

    class MultilingualWhisperSTT(OpenAISTTService):
        async def _transcribe(self, audio: bytes):
            from app.services.knowledge_service import knowledge_base

            domain_hint = knowledge_base.get_stt_hint()
            kwargs: dict = {
                "file":            ("audio.wav", audio, "audio/wav"),
                "model":           "whisper-1",
                "response_format": "verbose_json",   # gives .language + .segments[].no_speech_prob
            }
            if domain_hint:
                kwargs["prompt"] = domain_hint

            result = await self._client.audio.transcriptions.create(**kwargs)
            detected = getattr(result, "language", "") or ""

            # ── Suppress low-confidence / no-speech segments ─────────────────
            # verbose_json returns segments each with no_speech_prob (0-1).
            # If Whisper itself thinks this is mostly not speech, blank it out.
            segments = getattr(result, "segments", None) or []
            if segments:
                avg_no_speech = sum(
                    getattr(s, "no_speech_prob", 0) for s in segments
                ) / len(segments)
                if avg_no_speech > 0.55:
                    logger.info(
                        "STT suppressed (no_speech_prob={:.2f}): {!r}",
                        avg_no_speech, result.text,
                    )
                    try:
                        object.__setattr__(result, "text", "")
                    except Exception:
                        pass
                    return result

            logger.info(
                "STT: {!r}  lang={!r}  no_speech_prob={:.2f}",
                result.text, detected,
                sum(getattr(s,"no_speech_prob",0) for s in segments) / max(len(segments),1),
            )
            return result

    return MultilingualWhisperSTT(api_key=api_key)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

async def run_voice_pipeline(
    websocket: WebSocket,
    session_id: str,
    agent,
    db,
) -> None:

    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.frames.frames import (
            EndFrame, Frame, StartFrame, TTSSpeakFrame, TranscriptionFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.services.openai.tts import OpenAITTSService
        from pipecat.transports.websocket.fastapi import (
            FastAPIWebsocketParams, FastAPIWebsocketTransport,
        )
    except ImportError as exc:
        logger.error("pipecat-ai import failed: {}", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
        return

    api_key = os.getenv("OPENAI_API_KEY", "")

    # Track current session language so TTS instructions stay in sync
    _session_lang: dict[str, str] = {"lang": "ta"}

    # --- agent bridge -------------------------------------------------------
    class AgentFrameProcessor(FrameProcessor):

        async def _call_agent(self, text: str, detected_lang: str | None) -> dict:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, agent.handle, session_id, text, detected_lang
            )

        async def _send_meta(self, result: dict, transcript: str = "") -> None:
            try:
                await websocket.send_json({
                    "type":       "agent_data",
                    "transcript": transcript,
                    "reply":      result.get("reply", ""),
                    "lang":       result.get("lang", "ta"),
                    "emotion":    result.get("emotion", "neutral"),
                    "phase":      result.get("phase", ""),
                    "lead":       result.get("lead", {}),
                    "done":       result.get("done", False),
                })
            except Exception as exc:
                logger.warning("send_meta failed: {}", exc)

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            # ── Greeting ──────────────────────────────────────────────
            if isinstance(frame, StartFrame):
                await self.push_frame(frame, direction)
                logger.info("Pipeline started (session={})", session_id)
                try:
                    result = await self._call_agent("__start__", None)
                    await self._send_meta(result)
                    reply = _clean_for_tts(result.get("reply", ""))
                    if reply:
                        await self.push_frame(TTSSpeakFrame(text=reply), FrameDirection.DOWNSTREAM)
                except Exception as exc:
                    logger.error("Greeting failed: {}", exc)
                return

            # ── Transcribed user speech ───────────────────────────────
            if isinstance(frame, TranscriptionFrame) and frame.text.strip() and _is_valid_transcript(frame.text):
                transcript = frame.text.strip()

                # ── FAST language detection (two-stage) ─────────────────
                # Stage 1: Unicode script scan — zero latency
                detected_lang = _detect_script(transcript)

                # Stage 2: Whisper's own language detection (from verbose_json)
                #          Only needed if stage 1 returned None (Latin script)
                if detected_lang is None and frame.result is not None:
                    whisper_lang = getattr(frame.result, "language", "") or ""
                    detected_lang = _whisper_lang_to_code(whisper_lang)

                if detected_lang:
                    logger.info(
                        "Language detected: {} (script={}, whisper={})",
                        detected_lang,
                        _detect_script(transcript),
                        getattr(frame.result, "language", "?") if frame.result else "?",
                    )
                    # Update TTS instructions for this session immediately
                    if detected_lang != _session_lang["lang"]:
                        _session_lang["lang"] = detected_lang
                        new_instr = _get_tts_instructions(detected_lang)
                        try:
                            await tts._update_settings(
                                OpenAITTSService.Settings(instructions=new_instr)
                            )
                            logger.info("TTS instructions updated → {}", detected_lang)
                        except Exception as e:
                            logger.debug("TTS update_settings skipped: {}", e)

                logger.info("User said: {!r}  (lang={})", transcript, detected_lang or "?")

                try:
                    result = await self._call_agent(transcript, detected_lang)
                    logger.info("Agent reply: {!r}  (lang={})", result.get("reply",""), result.get("lang",""))
                    await self._send_meta(result, transcript)

                    # Keep session lang in sync with what the agent replied in
                    agent_lang = result.get("lang", "ta")
                    if agent_lang and agent_lang != _session_lang["lang"]:
                        _session_lang["lang"] = agent_lang

                    reply = _clean_for_tts(result.get("reply", ""))
                    if reply:
                        await self.push_frame(TTSSpeakFrame(text=reply), FrameDirection.DOWNSTREAM)
                    if result.get("done"):
                        db.save(result["lead"], session_id)
                        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
                except Exception as exc:
                    logger.error("Agent call failed: {}", exc)
                return

            await self.push_frame(frame, direction)

    # --- VAD ---------------------------------------------------------------
    # Tuned to prevent false triggers from background noise / TTS bleed-through:
    # • confidence 0.7  — require high model certainty before treating as speech
    # • start_secs 0.2  — 200 ms of real speech before triggering (ignores clicks)
    # • stop_secs  0.9  — 900 ms silence gap = natural end of sentence
    # • min_volume 0.4  — loud enough to be intentional speech (filters hiss)
    vad_params = VADParams(
        confidence=0.75,  # high certainty — rejects noise bursts
        start_secs=0.30,  # 300 ms of sustained speech before triggering (was 0.20)
        stop_secs=1.00,   # 1 s silence = finished speaking
        min_volume=0.45,  # intentional speech volume
    )

    # --- Transport ---------------------------------------------------------
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16_000,
            audio_out_enabled=True,
            audio_out_sample_rate=24_000,
            audio_out_10ms_chunks=20,
            audio_out_auto_silence=False,
            add_wav_header=True,
            serializer=RawPCMFrameSerializer(),
        ),
    )

    stt = _make_multilingual_stt(api_key)

    # TTS starts with Tamil instructions (default language)
    tts = OpenAITTSService(
        api_key=api_key,
        settings=OpenAITTSService.Settings(
            model="gpt-4o-mini-tts",
            voice="nova",
            instructions=_get_tts_instructions("ta"),
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params)),
        stt,
        AgentFrameProcessor(),
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        enable_rtvi=False,
    )
    runner = PipelineRunner(handle_sigint=False)

    logger.info("Voice pipeline starting  session={}", session_id)
    try:
        await runner.run(task)
    except Exception as exc:
        logger.error("Voice pipeline error  session={}: {}", session_id, exc, exc_info=True)
    finally:
        logger.info("Voice pipeline ended  session={}", session_id)
