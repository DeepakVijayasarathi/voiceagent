from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pathlib import Path
import yaml
import asyncio
import concurrent.futures
import logging

from app.services.agent_service import AgentService, set_knowledge_base, set_company_info, get_company_info
from app.services.knowledge_service import knowledge_base
from app.services.db_service import DBService
from app.services.audio_service import AudioService
from app.services.pipecat_service import run_voice_pipeline

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "agent.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _AGENT_CFG = yaml.safe_load(_f)["agent"]

app = FastAPI(title="Voice Sales Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AgentService()
db    = DBService()
audio = AudioService()

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
_POOL     = concurrent.futures.ThreadPoolExecutor()


# ---------------------------------------------------------------------------
# Startup — pre-warm filler cache
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_POOL, audio.prime_fillers)


# ---------------------------------------------------------------------------
# PDF upload — extract text, inject into agent knowledge base
# ---------------------------------------------------------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are accepted"})
    try:
        import pdfplumber, io, json as _json
        from openai import OpenAI as _OAI
        raw = await file.read()
        text_parts = []
        num_pages = 0
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        full_text = "\n".join(text_parts).strip()
        if not full_text:
            return JSONResponse(status_code=422, content={"error": "No readable text found in PDF"})

        # ── 1. Store raw text + build RAG index (embeddings) ──────────
        set_knowledge_base(full_text)
        chunks_n = await asyncio.get_event_loop().run_in_executor(
            _POOL, knowledge_base.ingest, full_text
        )
        log.info("RAG index built: %d chunks", chunks_n)

        # ── 2. Extract company info via GPT ────────────────────────────
        try:
            _client = _OAI(api_key=_AGENT_CFG.get("llm", {}).get("api_key") or __import__("os").getenv("OPENAI_API_KEY"))
            resp = await asyncio.get_event_loop().run_in_executor(
                _POOL,
                lambda: _client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=250,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract company details from this business document. "
                                "Return ONLY valid JSON with these exact keys:\n"
                                '{"name":"<company name>","tagline":"<short tagline or description>","agent_name":"<sales agent name, default Priya>","services":"<comma-separated list of products or services>","location":"<city or location if mentioned>"}'
                            ),
                        },
                        {"role": "user", "content": full_text[:4000]},
                    ],
                    response_format={"type": "json_object"},
                ),
            )
            company_info = _json.loads(resp.choices[0].message.content)
            set_company_info(company_info)
            log.info("Company info extracted from PDF: %s", company_info)
        except Exception as exc:
            log.warning("Company info extraction failed (using defaults): %s", exc)

        preview = full_text[:200].replace("\n", " ")
        return {
            "status":  "ok",
            "pages":   num_pages,
            "chars":   len(full_text),
            "chunks":  chunks_n,
            "preview": preview,
            "company": get_company_info(),
        }
    except ImportError:
        return JSONResponse(status_code=500,
                            content={"error": "pdfplumber not installed — run: pip install pdfplumber"})
    except Exception as exc:
        log.error("PDF upload error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Filler audio  (OpenAI MP3, cached)
# ---------------------------------------------------------------------------

@app.get("/filler/{lang}")
async def get_filler(lang: str):
    data = await asyncio.get_event_loop().run_in_executor(
        _POOL, audio.get_filler, lang
    )
    return Response(content=data, media_type="audio/mpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: Request):
    data       = await req.json()
    session_id = data.get("session_id", "default")
    message    = data.get("message", "")

    result = await asyncio.get_event_loop().run_in_executor(
        _POOL, agent.handle, session_id, message
    )
    if result["done"]:
        db.save(result["lead"], session_id)
    return result


# ---------------------------------------------------------------------------
# Speech-to-Text  (Whisper)
# ---------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes  = await file.read()
    content_type = file.content_type or "audio/webm"
    text = await asyncio.get_event_loop().run_in_executor(
        _POOL, audio.transcribe, audio_bytes, content_type
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Text-to-Speech  (ElevenLabs → OpenAI, REST fallback mode)
# ---------------------------------------------------------------------------

@app.post("/tts")
async def tts(req: Request):
    body = await req.json()
    text = body.get("text", "").strip()
    lang = body.get("lang", "ta")
    if not text:
        return Response(status_code=400)

    data, ctype, is_stream = await asyncio.get_event_loop().run_in_executor(
        _POOL, audio.speak, text, lang
    )

    if is_stream:
        return StreamingResponse(data, media_type=ctype)
    return Response(content=data, media_type=ctype)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@app.get("/config")
async def get_config():
    co = get_company_info()   # reflects PDF-extracted info once a PDF is uploaded
    return {
        "company":    co.get("name",       "Sales Agent"),
        "tagline":    co.get("tagline",    ""),
        "agent_name": co.get("agent_name", "Priya"),
        "location":   co.get("location",   ""),
    }


# ---------------------------------------------------------------------------
# Leads + session
# ---------------------------------------------------------------------------

@app.get("/leads")
async def get_leads():
    return db.get_all()


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    agent.clear_session(session_id)
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Pipecat real-time voice  (WebSocket)
# ---------------------------------------------------------------------------

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket, session_id: str = "default"):
    await websocket.accept()
    try:
        await run_voice_pipeline(websocket, session_id, agent, db)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected (session=%s)", session_id)
    except Exception as exc:
        log.error("Unhandled voice pipeline error (session=%s): %s", session_id, exc)


# ---------------------------------------------------------------------------
# Frontend  (must be last)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
