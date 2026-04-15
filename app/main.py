from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pathlib import Path
import yaml
import asyncio
import concurrent.futures
import logging

# ---------------------------------------------------------------------------
# Phoenix tracing — instrument OpenAI calls
# ---------------------------------------------------------------------------
import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as _otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as _otel_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_phoenix_session = px.launch_app()
_tracer_provider = _otel_sdk.TracerProvider()
_tracer_provider.add_span_processor(
    SimpleSpanProcessor(
        OTLPSpanExporter(endpoint=f"{_phoenix_session.url}v1/traces")
    )
)
_otel_trace.set_tracer_provider(_tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)

from app.services.agent_service import AgentService, get_company_info
from app.services.tenant_service import TenantRegistry
from app.services.db_service import DBService
from app.services.audio_service import AudioService
from app.services.pipecat_service import run_voice_pipeline

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "agent.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _FULL_CFG = yaml.safe_load(_f)

_AGENT_CFG = _FULL_CFG.get("agent", {})

app = FastAPI(title="Voice Sales Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = TenantRegistry(_FULL_CFG)
agent    = AgentService()
db       = DBService()
audio    = AudioService()

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
# PDF upload — extract text, inject into tenant knowledge base
# ---------------------------------------------------------------------------

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    tenant_id: str = Query(default="default", description="Tenant to load the PDF into"),
):
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

        tenant = registry.get(tenant_id)

        # ── 1. Store raw text + build RAG index (embeddings) ──────────
        tenant.pdf_text = full_text
        chunks_n = await asyncio.get_event_loop().run_in_executor(
            _POOL, tenant.knowledge_base.ingest, full_text
        )
        log.info("RAG index built: %d chunks (tenant=%s)", chunks_n, tenant_id)

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
            registry.update_company(tenant_id, company_info)
            log.info("Company info extracted (tenant=%s): %s", tenant_id, company_info)
        except Exception as exc:
            log.warning("Company info extraction failed (using defaults): %s", exc)

        preview = full_text[:200].replace("\n", " ")
        return {
            "status":    "ok",
            "tenant_id": tenant_id,
            "pages":     num_pages,
            "chars":     len(full_text),
            "chunks":    chunks_n,
            "preview":   preview,
            "company":   get_company_info(tenant),
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
    tenant_id  = data.get("tenant_id", "default")
    message    = data.get("message", "")

    tenant = registry.get(tenant_id)
    result = await asyncio.get_event_loop().run_in_executor(
        _POOL, agent.handle, session_id, message, None, tenant
    )
    if result["done"]:
        db.save(result["lead"], session_id, tenant_id)
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
async def get_config(
    tenant_id: str = Query(default="default", description="Tenant ID"),
):
    tenant = registry.get(tenant_id)
    co = get_company_info(tenant)
    return {
        "tenant_id":  tenant_id,
        "company":    co.get("name",       "Sales Agent"),
        "tagline":    co.get("tagline",    ""),
        "agent_name": co.get("agent_name", "Priya"),
        "location":   co.get("location",   ""),
    }


# ---------------------------------------------------------------------------
# Leads + session
# ---------------------------------------------------------------------------

@app.get("/leads")
async def get_leads(
    tenant_id: str = Query(default="default", description="Tenant ID"),
):
    return db.get_all(tenant_id)


@app.get("/tenants")
async def list_tenants():
    return {"tenants": registry.list_tenants()}


@app.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    tenant_id: str = Query(default="default", description="Tenant ID"),
):
    agent.clear_session(session_id, tenant_id)
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Pipecat real-time voice  (WebSocket)
# ---------------------------------------------------------------------------

@app.websocket("/ws/voice")
async def voice_websocket(
    websocket: WebSocket,
    session_id: str = "default",
    tenant_id: str = "default",
):
    await websocket.accept()
    tenant = registry.get(tenant_id)
    try:
        await run_voice_pipeline(websocket, session_id, agent, db, tenant)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected (session=%s, tenant=%s)", session_id, tenant_id)
    except Exception as exc:
        log.error("Unhandled voice pipeline error (session=%s): %s", session_id, exc)


# ---------------------------------------------------------------------------
# Frontend  (must be last)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
