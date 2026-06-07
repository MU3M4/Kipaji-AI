"""
main.py — Kipaji Core AI v3.1 (Groq Llama 3.3 Edition)
AI-driven cash-velocity credit engine for African informal economy MSMEs
"""
import os
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Local modules
from agents import (
    run_bias_mitigation_guardrail,
    run_kipaji_underwriter_core,
    SanitizedInput,
    CreditDecision,
)
from storage import (
    get_merchant_data,
    save_trade_interaction,
    get_ussd_session,
    save_ussd_session,
    clear_ussd_session,
)

# Language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("KipajiCoreAI")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS    = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------
ai_client = None
try:
    if GROQ_API_KEY:
        from openai import OpenAI
        # Groq is OpenAI-compatible. We just point the base_url to Groq.
        ai_client = OpenAI(
            api_key=GROQ_API_KEY, 
            base_url="https://api.groq.com/openai/v1"
        )
        logger.info("Groq AI client initialised")
    else:
        logger.warning("GROQ_API_KEY missing — rule-based fallback will be used")
except Exception as e:
    logger.error(f"Groq client init failed: {e}")

db = None
try:
    from google.cloud import firestore
    db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
    logger.info("Firestore initialised")
except Exception as e:
    logger.warning(f"Firestore unavailable — in-memory ledger active: {e}")

redis_client = None
try:
    import redis as redis_lib
    redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
    redis_client.ping()
    logger.info("Redis initialised")
except Exception as e:
    logger.warning(f"Redis unavailable — in-memory USSD sessions active: {e}")

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kipaji Core AI",
    description="AI-driven cash-velocity credit engine for African informal economy MSMEs",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# TELEMETRY HUB (WebSocket — fire-and-forget, never on critical path)
# ---------------------------------------------------------------------------
class TelemetryHub:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Telemetry client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Telemetry client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
        payload = json.dumps(message, default=str)
        dead = []
        for conn in self.active_connections[:]:
            try:
                await conn.send_text(payload)
            except Exception:
                dead.append(conn)
        for ws in dead:
            self.disconnect(ws)

telemetry_hub = TelemetryHub()

async def _emit_audit(event_type: str, merchant_id: str, data: Dict[str, Any]):
    try:
        await telemetry_hub.broadcast({
            "event": event_type,
            "merchant_id": merchant_id,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        })
    except Exception as e:
        logger.debug(f"Telemetry emit error (non-critical): {e}")

# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------
class GatewayMessage(BaseModel):
    merchant_id: str = Field(..., min_length=3, max_length=64)
    phone_number: str = Field(..., pattern=r"^+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1, max_length=2000)

class USSDRequest(BaseModel):
    sessionId: str
    serviceCode: str
    phoneNumber: str
    text: str

class USSDResponse(BaseModel):
    response: str

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
_SWAHILI_MARKERS = {"nimeuza", "niliuza", "nilinunua", "biashara", "bei", "leo", "jana", "shilingi", "pesa", "faida", "hasara", "mteja", "wanunuzi", "soko", "nimelipa", "ninapata", "shs", "ksh"}
_DHOLUO_MARKERS = {"aora", "chwo", "nying", "gima", "pesa", "chiel", "ariyo", "adek", "ochiko", "udhuro", "dhano", "ngadi", "ohala"}
_SHENG_MARKERS = {"niko na", "nimefanya", "nilifanya", "ilikuwa", "imekuwa", "chapaa", "mkwanja", "mdo", "ka", "fiti"}

def detect_language(text: str) -> str:
    tokens = set(text.lower().split())
    if tokens & _DHOLUO_MARKERS: return "luo"
    if tokens & _SWAHILI_MARKERS: return "sw"
    if tokens & _SHENG_MARKERS: return "sw"
    try:
        detected = detect(text[:400])
        return "sw" if detected != "en" else "en"
    except Exception:
        return "sw"

def calculate_velocity_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    sales = [
        t for t in history
        if t.get("type") in ("revenue", "sale_entry_parsed")
        and isinstance(t.get("amount_local"), (int, float))
        and t.get("amount_local", 0) > 0
    ]
    if not sales:
        return {"avg_daily": 0.0, "total_7d": 0.0, "consistency": 1.0, "transaction_count": 0}

    recent = sales[-7:]
    amounts = [float(t["amount_local"]) for t in recent]
    total_7d = sum(amounts)
    avg_daily = total_7d / 7.0

    if len(amounts) > 1:
        mean = total_7d / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std = variance ** 0.5
        consistency = max(0.0, 1.0 - (std / (mean + 1)))
    else:
        consistency = 0.5

    return {
        "avg_daily": round(avg_daily, 2),
        "total_7d": round(total_7d, 2),
        "consistency": round(consistency, 2),
        "transaction_count": len(sales),
    }

def _extract_primary_trade_amount(sanitized: SanitizedInput) -> float:
    revenue_events = [
        e for e in sanitized.detected_trade_events
        if e.event_type in ("revenue", "receivable")
        and e.amount_ksh is not None
        and e.confidence >= 0.5
    ]
    if not revenue_events: return 0.0
    best = max(revenue_events, key=lambda e: e.confidence)
    return best.amount_ksh

# ---------------------------------------------------------------------------
# WEBSOCKET TELEMETRY ENDPOINT
# ---------------------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_hub.disconnect(websocket)

# ---------------------------------------------------------------------------
# MAIN GATEWAY — WhatsApp / SMS / API
# ---------------------------------------------------------------------------
@app.post("/api/v1/gateway")
async def inbound_telecom_gateway(payload: GatewayMessage, background_tasks: BackgroundTasks):
    merchant_id = payload.merchant_id
    logger.info(f"[{merchant_id}] Inbound via {payload.channel}: {payload.message_body[:80]}...")

    user_lang = detect_language(payload.message_body)
    merchant_data = await get_merchant_data(merchant_id, db=db)
    velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

    sanitized = await run_bias_mitigation_guardrail(
        raw_message=payload.message_body, merchant_id=merchant_id, language=user_lang, ai_client=ai_client
    )
    decision = await run_kipaji_underwriter_core(
        sanitized=sanitized, history=merchant_data["history"], profile=merchant_data["profile"],
        velocity_metrics=velocity_metrics, merchant_id=merchant_id, ai_client=ai_client
    )

    extracted_amount = _extract_primary_trade_amount(sanitized)
    primary_event_type = sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"
    
    background_tasks.add_task(
        save_trade_interaction, merchant_id=merchant_id, extracted_trade_amount=extracted_amount,
        event_type=primary_event_type, decision_dict=decision.model_dump(),
        bias_removed=sanitized.bias_proxy_removed, language=user_lang, db=db
    )
    background_tasks.add_task(
        _emit_audit, "credit_decision", merchant_id, {
            "channel": payload.channel, "language": user_lang, "credit_tier": decision.credit_tier,
            "approved_local": decision.approved_local, "velocity_score": decision.velocity_score,
            "confidence_gate_triggered": decision.confidence_gate_triggered,
            "bias_proxy_removed": sanitized.bias_proxy_removed, "audit_trail": decision.audit_trail
        }
    )

    return {
        "status": "SUCCESS", "merchant_id": merchant_id, "language_detected": user_lang,
        "credit_decision": decision.model_dump(), "velocity_metrics": velocity_metrics,
        "response_message": decision.response_message,
    }

# ---------------------------------------------------------------------------
# USSD GATEWAY
# ---------------------------------------------------------------------------
@app.post("/api/v1/ussd", response_model=None)
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    session_id   = form.get("sessionId", "")
    phone_number = form.get("phoneNumber", "")
    text         = form.get("text", "")

    merchant_id = "m_" + phone_number.replace("+", "").replace(" ", "")
    logger.info(f"[USSD] session={session_id} phone={phone_number} text='{text}'")

    session = await get_ussd_session(session_id, redis_client=redis_client)
    if session is None:
        session = {"step": 0, "merchant_id": merchant_id, "phone": phone_number}

    step = session.get("step", 0)

    if text == "" or step == 0:
        session["step"] = 1
        await save_ussd_session(session_id, session, redis_client=redis_client)
        return _ussd_continue("Karibu Kipaji!\nTuambie biashara yako ya leo.\nMfano: Niliwauzia wateja mchele 3kg kwa 150 kila moja\n\nAndika ujumbe wako:")

    if step == 1:
        trade_message = text.strip()
        if len(trade_message) < 5:
            return _ussd_end("Ujumbe mfupi sana. Tafadhali jaribu tena. *384#")

        user_lang = detect_language(trade_message)
        merchant_data = await get_merchant_data(merchant_id, db=db)
        velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

        sanitized = await run_bias_mitigation_guardrail(trade_message, merchant_id, user_lang, ai_client)
        decision = await run_kipaji_underwriter_core(
            sanitized, merchant_data["history"], merchant_data["profile"],
            velocity_metrics, merchant_id, ai_client
        )

        if decision.confidence_gate_triggered:
            session["step"] = 2
            session["partial_message"] = trade_message
            await save_ussd_session(session_id, session, redis_client=redis_client)
            return _ussd_continue(decision.response_message[:160] + "\n\nJibu:")

        extracted_amount = _extract_primary_trade_amount(sanitized)
        background_tasks.add_task(
            save_trade_interaction, merchant_id, extracted_amount,
            (sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"),
            decision.model_dump(), sanitized.bias_proxy_removed, user_lang, db
        )
        background_tasks.add_task(_emit_audit, "ussd_credit_decision", merchant_id, {"tier": decision.credit_tier, "approved_local": decision.approved_local})

        await clear_ussd_session(session_id, redis_client=redis_client)
        return _ussd_end(decision.response_message[:160])

    if step == 2:
        combined = session.get("partial_message", "") + " " + text.strip()
        session["step"] = 1
        session["partial_message"] = ""
        
        user_lang = detect_language(combined)
        merchant_data = await get_merchant_data(merchant_id, db=db)
        velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

        sanitized = await run_bias_mitigation_guardrail(combined, merchant_id, user_lang, ai_client)
        decision = await run_kipaji_underwriter_core(
            sanitized, merchant_data["history"], merchant_data["profile"],
            velocity_metrics, merchant_id, ai_client
        )

        extracted_amount = _extract_primary_trade_amount(sanitized)
        background_tasks.add_task(
            save_trade_interaction, merchant_id, extracted_amount,
            (sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"),
            decision.model_dump(), sanitized.bias_proxy_removed, user_lang, db
        )

        await clear_ussd_session(session_id, redis_client=redis_client)
        return _ussd_end(decision.response_message[:160])

    await clear_ussd_session(session_id, redis_client=redis_client)
    return _ussd_end("Kuna tatizo. Tafadhali piga *384# tena.")

def _ussd_continue(text: str) -> str: return f"CON {text}"
def _ussd_end(text: str) -> str: return f"END {text}"

# ---------------------------------------------------------------------------
# UTILITY ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "service": "Kipaji Core AI", "version": "3.1.0", "status": "ONLINE",
        "groq": "connected" if ai_client else "unavailable (rule-based fallback active)",
        "firestore": "connected" if db else "unavailable (in-memory fallback active)",
        "redis": "connected" if redis_client else "unavailable (in-memory sessions active)",
    }

@app.get("/health")
async def health_check():
    checks = {"api": True, "groq_configured": bool(GROQ_API_KEY), "firestore": False, "redis": False}
    if db:
        try:
            await asyncio.to_thread(lambda: db.collection("_health").document("ping").get())
            checks["firestore"] = True
        except Exception: pass
    if redis_client:
        try:
            await asyncio.to_thread(redis_client.ping)
            checks["redis"] = True
        except Exception: pass
    overall = checks["api"] and checks["groq_configured"]
    return {"healthy": overall, "checks": checks, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v1/merchant/{merchant_id}/history")
async def get_merchant_history(merchant_id: str):
    data = await get_merchant_data(merchant_id, db=db)
    metrics = calculate_velocity_metrics(data["history"])
    return {
        "merchant_id": merchant_id, "trade_history_count": len(data["history"]),
        "recent_history": data["history"][-10:], "velocity_metrics": metrics, "profile": data["profile"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=os.getenv("ENV", "production") == "development", log_level="info")