"""
main.py — Kipaji Core AI v3.1 (Groq Edition)
AI-driven cash-velocity credit engine for African informal economy MSMEs

Architecture:
FastAPI gateway → Agent 1 (bias guardrail) → Agent 2 (underwriter) → credit decision
WebSocket telemetry hub (async, fire-and-forget — never blocks credit path)
Redis session state for USSD continuity
In-memory ledger (Firestore removed for hackathon stability)

Key Features:
- Groq Llama 3.3 integration (bypasses Google billing wall)
- Typed inter-agent contract (Pydantic)
- Confidence gate (routes to clarification below 0.65)
- Deterministic rule-based fallback (zero single point of failure)
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
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS    = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------
# 1. AI Client (Groq / OpenAI compatible)
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

# 2. Database (Explicitly None to force in-memory fallback for hackathon)
db = None

# 3. Redis (For USSD sessions)
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
# TELEMETRY HUB  (WebSocket — fire-and-forget, never on critical path)
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
        """Non-blocking broadcast — silently drops if no consumers."""
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
    """Fire-and-forget telemetry emission. Swallows all errors."""
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
    phone_number: str = Field(..., pattern=r"^\+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1, max_length=2000)

class USSDRequest(BaseModel):
    """Africa's Talking / Safaricom USSD gateway format."""
    sessionId: str
    serviceCode: str
    phoneNumber: str
    text: str                   # accumulates across menus e.g. "12500"

class USSDResponse(BaseModel):
    response: str               # prefixed with CON (continue) or END (terminate)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
# Lexical markers for informal East African language variants
_SWAHILI_MARKERS = {
    "nimeuza", "niliuza", "nilinunua", "biashara", "bei", "leo", "jana",
    "shilingi", "pesa", "faida", "hasara", "mteja", "wanunuzi", "soko",
    "nimelipa", "ninapata", "shs", "ksh",
}
_DHOLUO_MARKERS = {
    "aora", "chwo", "nying", "gima", "pesa", "chiel", "ariyo", "adek",
    "ochiko", "udhuro", "dhano", "ngadi", "ohala",
}
_SHENG_MARKERS = {
    "niko na", "nimefanya", "nilifanya", "ilikuwa", "imekuwa", "chapaa",
    "mkwanja", "mdo", "ka", "fiti",
}

def detect_language(text: str) -> str:
    """
    Lexical pre-check (zero latency) before falling back to langdetect.
    Handles Swahili, Dholuo, Sheng, and English reliably.
    """
    tokens = set(text.lower().split())
    if tokens & _DHOLUO_MARKERS:
        return "luo"
    if tokens & _SWAHILI_MARKERS:
        return "sw"
    if tokens & _SHENG_MARKERS:
        return "sw"   # treat Sheng as Swahili for model purposes

    try:
        detected = detect(text[:400])
        if detected in ("sw", "en", "so", "lg"):
            return "sw" if detected != "en" else "en"
        return "sw"   # default for unrecognised East African languages
    except Exception:
        return "sw"

def calculate_velocity_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes 7-day cash-velocity metrics from merchant trade history.
    FIX: avg_daily always divides by 7 (the time window), not by transaction count.
    """
    sales = [
        t for t in history
        if t.get("type") in ("revenue", "sale_entry_parsed")
        and isinstance(t.get("amount_local"), (int, float))
        and t.get("amount_local", 0) > 0
    ]

    if not sales:
        return {"avg_daily": 0.0, "total_7d": 0.0, "consistency": 1.0, "transaction_count": 0}

    recent = sales[-7:]   # at most the last 7 entries
    amounts = [float(t["amount_local"]) for t in recent]
    total_7d = sum(amounts)

    # Always divide by 7 days — not by number of transactions
    avg_daily = total_7d / 7.0

    # Consistency: 1.0 = perfectly even, 0.0 = extremely volatile
    if len(amounts) > 1:
        mean = total_7d / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std = variance ** 0.5
        consistency = max(0.0, 1.0 - (std / (mean + 1)))
    else:
        consistency = 0.5   # single data point — neutral, not perfect

    return {
        "avg_daily": round(avg_daily, 2),
        "total_7d": round(total_7d, 2),
        "consistency": round(consistency, 2),
        "transaction_count": len(sales),
    }

def _extract_primary_trade_amount(sanitized: SanitizedInput) -> float:
    """
    Extract the primary revenue amount from the guardrail output.
    This is what gets stored in the ledger — NOT the approved credit amount.
    """
    revenue_events = [
        e for e in sanitized.detected_trade_events
        if e.event_type in ("revenue", "receivable")
        and e.amount_ksh is not None
        and e.confidence >= 0.5
    ]
    if not revenue_events:
        return 0.0
    # Use highest-confidence event if multiple
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
            # Keep alive — we only push from server to client
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_hub.disconnect(websocket)

# ---------------------------------------------------------------------------
# MAIN GATEWAY — WhatsApp / SMS / API
# ---------------------------------------------------------------------------
@app.post("/api/v1/gateway")
async def inbound_telecom_gateway(
    payload: GatewayMessage,
    background_tasks: BackgroundTasks,
):
    """
    Primary credit gateway for WhatsApp, SMS, and direct API channels.
    Pipeline:
    1. Language detection (lexical + langdetect)
    2. Merchant data retrieval (In-memory)
    3. Velocity metrics calculation
    4. Agent 1: Bias mitigation guardrail
    5. Agent 2: Kipaji underwriter (with confidence gate)
    6. Async: persist trade event to ledger
    7. Async: emit audit event to telemetry hub
    8. Return credit decision to caller
    """
    merchant_id = payload.merchant_id
    logger.info(f"[{merchant_id}] Inbound via {payload.channel}: {payload.message_body[:80]}...")

    # Step 1: Language
    user_lang = detect_language(payload.message_body)

    # Step 2: Merchant data (db=None forces in-memory fallback)
    merchant_data = await get_merchant_data(merchant_id)

    # Step 3: Velocity metrics
    velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

    # Step 4: Bias guardrail (Agent 1)
    sanitized = await run_bias_mitigation_guardrail(
        raw_message=payload.message_body,
        merchant_id=merchant_id,
        language=user_lang,
        ai_client=ai_client,
    )

    # Step 5: Underwriter (Agent 2) — confidence gate is inside this call
    decision = await run_kipaji_underwriter_core(
        sanitized=sanitized,
        history=merchant_data["history"],
        profile=merchant_data["profile"],
        velocity_metrics=velocity_metrics,
        merchant_id=merchant_id,
        ai_client=ai_client,
    )

    # Step 6: Persist trade event (background — non-blocking)
    extracted_amount = _extract_primary_trade_amount(sanitized)
    primary_event_type = (
        sanitized.detected_trade_events[0].event_type
        if sanitized.detected_trade_events else "unknown"
    )
    background_tasks.add_task(
        save_trade_interaction,
        merchant_id=merchant_id,
        extracted_trade_amount=extracted_amount,
        event_type=primary_event_type,
        decision_dict=decision.model_dump(),
        bias_removed=sanitized.bias_proxy_removed,
        language=user_lang,
    )

    # Step 7: Telemetry audit (background — fire-and-forget)
    background_tasks.add_task(
        _emit_audit,
        event_type="credit_decision",
        merchant_id=merchant_id,
        data={
            "channel": payload.channel,
            "language": user_lang,
            "credit_tier": decision.credit_tier,
            "approved_local": decision.approved_local,
            "velocity_score": decision.velocity_score,
            "confidence_gate_triggered": decision.confidence_gate_triggered,
            "bias_proxy_removed": sanitized.bias_proxy_removed,
            "audit_trail": decision.audit_trail,
        },
    )

    # Step 8: Response
    return {
        "status": "SUCCESS",
        "merchant_id": merchant_id,
        "language_detected": user_lang,
        "credit_decision": decision.model_dump(),
        "velocity_metrics": velocity_metrics,
        "response_message": decision.response_message,
    }

# ---------------------------------------------------------------------------
# USSD GATEWAY — Telecom integration (Africa's Talking / Safaricom format)
# ---------------------------------------------------------------------------
@app.post("/api/v1/ussd", response_model=None)
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
    """
    USSD session handler compatible with Africa's Talking and Safaricom USSD gateway.
    Session state is persisted in Redis (TTL=180s mirrors network timeout).
    """
    # Africa's Talking sends form data, not JSON
    form = await request.form()
    session_id   = form.get("sessionId", "")
    phone_number = form.get("phoneNumber", "")
    text         = form.get("text", "")    # accumulates: "1*500*maize" style

    # Derive merchant_id from phone number (strip + prefix)
    merchant_id = "m_" + phone_number.replace("+", "").replace(" ", "")

    logger.info(f"[USSD] session={session_id} phone={phone_number} text='{text}'")

    # Retrieve or create session
    session = await get_ussd_session(session_id, redis_client=redis_client)
    if session is None:
        session = {"step": 0, "merchant_id": merchant_id, "phone": phone_number}

    step = session.get("step", 0)

    # ── Step 0: Welcome screen ──
    if text == "" or step == 0:
        session["step"] = 1
        await save_ussd_session(session_id, session, redis_client=redis_client)
        return _ussd_continue(
            "Karibu Kipaji!\n"
            "Tuambie biashara yako ya leo.\n"
            "Mfano: Niliwauzia wateja mchele 3kg kwa 150 kila moja\n\n"
            "Andika ujumbe wako: "
        )

    # ── Step 1: Collect trade message and process ──
    if step == 1:
        trade_message = text.strip()
        if len(trade_message) < 5:
            return _ussd_end("Ujumbe mfupi sana. Tafadhali jaribu tena. *384#")

        user_lang = detect_language(trade_message)
        merchant_data = await get_merchant_data(merchant_id)
        velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

        # Run the full AI pipeline
        sanitized = await run_bias_mitigation_guardrail(
            raw_message=trade_message,
            merchant_id=merchant_id,
            language=user_lang,
            ai_client=ai_client,
        )
        decision = await run_kipaji_underwriter_core(
            sanitized=sanitized,
            history=merchant_data["history"],
            profile=merchant_data["profile"],
            velocity_metrics=velocity_metrics,
            merchant_id=merchant_id,
            ai_client=ai_client,
        )

        # If confidence gate triggered, ask for clarification
        if decision.confidence_gate_triggered:
            session["step"] = 2
            session["partial_message"] = trade_message
            await save_ussd_session(session_id, session, redis_client=redis_client)
            # Truncate to 160 chars (USSD screen limit)
            clarification = decision.response_message[:160]
            return _ussd_continue(clarification + "\n\nJibu: ")

        # Persist and return
        extracted_amount = _extract_primary_trade_amount(sanitized)
        background_tasks.add_task(
            save_trade_interaction,
            merchant_id=merchant_id,
            extracted_trade_amount=extracted_amount,
            event_type=(sanitized.detected_trade_events[0].event_type
                        if sanitized.detected_trade_events else "unknown"),
            decision_dict=decision.model_dump(),
            bias_removed=sanitized.bias_proxy_removed,
            language=user_lang,
        )
        background_tasks.add_task(
            _emit_audit, "ussd_credit_decision", merchant_id,
            {"tier": decision.credit_tier, "approved_local": decision.approved_local},
        )

        await clear_ussd_session(session_id, redis_client=redis_client)
        result_msg = decision.response_message[:160]
        return _ussd_end(result_msg)

    # ─ Step 2: Handle clarification response ──
    if step == 2:
        combined = session.get("partial_message", "") + " " + text.strip()
        session["step"] = 1
        session["partial_message"] = ""
        
        # Re-process with combined context
        user_lang = detect_language(combined)
        merchant_data = await get_merchant_data(merchant_id)
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
            decision.model_dump(), sanitized.bias_proxy_removed, user_lang,
        )

        await clear_ussd_session(session_id, redis_client=redis_client)
        return _ussd_end(decision.response_message[:160])

    # Fallback
    await clear_ussd_session(session_id, redis_client=redis_client)
    return _ussd_end("Kuna tatizo. Tafadhali piga *384# tena.")

def _ussd_continue(text: str) -> str:
    """CON prefix tells the USSD gateway to keep the session open."""
    return f"CON {text}"

def _ussd_end(text: str) -> str:
    """END prefix tells the USSD gateway to close the session."""
    return f"END {text}"

# ---------------------------------------------------------------------------
# UTILITY ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "service": "Kipaji Core AI",
        "version": "3.1.0",
        "status": "ONLINE",
        "groq": "connected" if ai_client else "unavailable (rule-based fallback active)",
        "firestore": "disabled (in-memory fallback active)",
        "redis": "connected" if redis_client else "unavailable (in-memory sessions active)",
    }

@app.get("/health")
async def health_check():
    """Lightweight health check for uptime monitoring and judge demos."""
    checks = {
        "api": True,
        "groq_configured": bool(GROQ_API_KEY),
        "firestore": False,
        "redis": False,
    }
    if redis_client:
        try:
            await asyncio.to_thread(redis_client.ping)
            checks["redis"] = True
        except Exception:
            pass
    
    overall = checks["api"] and checks["groq_configured"]
    return {"healthy": overall, "checks": checks, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v1/merchant/{merchant_id}/history")
async def get_merchant_history(merchant_id: str):
    """Retrieve a merchant's trade history and velocity metrics. Useful for demo dashboard."""
    data = await get_merchant_data(merchant_id)
    metrics = calculate_velocity_metrics(data["history"])
    return {
        "merchant_id": merchant_id,
        "trade_history_count": len(data["history"]),
        "recent_history": data["history"][-10:],
        "velocity_metrics": metrics,
        "profile": data["profile"],
    }

# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )