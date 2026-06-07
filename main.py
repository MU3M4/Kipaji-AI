"""
main.py — Kipaji Core AI v4.0 (Production-Ready Edition)
AI-driven cash-velocity credit engine for African informal economy MSMEs

Architecture:
FastAPI gateway → Agent 1 (bias guardrail) → Agent 2 (underwriter) → credit decision
WebSocket telemetry hub with history cache (fire-and-forget — never blocks credit path)
Redis session state for USSD continuity (with in-memory fallback)
Firestore ledger with in-memory fallback
Live dashboard with real-time bias mitigation visualization

Fixes applied:
- USSD responses use PlainTextResponse (fixes "failed to launch" error)
- /health endpoint accepts HEAD requests (fixes UptimeRobot monitoring)
- Telemetry history cache (dashboard loads past decisions on reconnect)
- Judge audit endpoint (/api/v1/decisions)
- python-multipart support for USSD form parsing
"""
import os
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------
ai_client = None
try:
    if GEMINI_API_KEY:
        from google import genai
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI client initialised")
    else:
        logger.warning("GEMINI_API_KEY missing — rule-based fallback will be used")
except Exception as e:
    logger.error(f"Gemini client init failed: {e}")

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
    version="4.0.0",
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
# TELEMETRY HUB (WebSocket with History Cache)
# ---------------------------------------------------------------------------
telemetry_history = []
MAX_HISTORY = 20

class TelemetryHub:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Telemetry client connected. Total: {len(self.active_connections)}")
        
        # Send history to new client immediately
        for msg in telemetry_history:
            await websocket.send_text(json.dumps(msg, default=str))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Telemetry client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Non-blocking broadcast with history cache."""
        # Save to history
        telemetry_history.append(message)
        if len(telemetry_history) > MAX_HISTORY:
            telemetry_history.pop(0)
        
        # Broadcast to live connections
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
        msg = {
            "event": event_type,
            "merchant_id": merchant_id,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }
        logger.info(f"[Telemetry] Broadcasting: {msg}")
        await telemetry_hub.broadcast(msg)
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
    text: str

class USSDResponse(BaseModel):
    response: str

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
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
        return "sw"

    try:
        detected = detect(text[:400])
        if detected in ("sw", "en", "so", "lg"):
            return "sw" if detected != "en" else "en"
        return "sw"
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
async def inbound_telecom_gateway(
    payload: GatewayMessage,
    background_tasks: BackgroundTasks,
):
    """
    Primary credit gateway for WhatsApp, SMS, and direct API channels.
    """
    merchant_id = payload.merchant_id
    logger.info(f"[{merchant_id}] Inbound via {payload.channel}: {payload.message_body[:80]}...")

    user_lang = detect_language(payload.message_body)
    merchant_data = await get_merchant_data(merchant_id, db=db)
    velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

    sanitized = await run_bias_mitigation_guardrail(
        raw_message=payload.message_body,
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
        db=db,
    )

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

    return {
        "status": "SUCCESS",
        "merchant_id": merchant_id,
        "language_detected": user_lang,
        "credit_decision": decision.model_dump(),
        "velocity_metrics": velocity_metrics,
        "response_message": decision.response_message,
    }

# ---------------------------------------------------------------------------
# USSD GATEWAY — Telecom integration (FIXED: PlainTextResponse)
# ---------------------------------------------------------------------------
@app.post("/api/v1/ussd", response_model=None)
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
    """
    USSD session handler compatible with Africa's Talking and Safaricom USSD gateway.
    Session state is persisted in Redis (TTL=180s mirrors network timeout).
    """
    form = await request.form()
    session_id = form.get("sessionId", "")
    phone_number = form.get("phoneNumber", "")
    text = form.get("text", "")

    merchant_id = "m_" + phone_number.replace("+", "").replace(" ", "")
    logger.info(f"[USSD] session={session_id} phone={phone_number} text='{text}'")

    session = await get_ussd_session(session_id, redis_client=redis_client)
    if session is None:
        session = {"step": 0, "merchant_id": merchant_id, "phone": phone_number}

    step = session.get("step", 0)

    if text == "" or step == 0:
        session["step"] = 1
        await save_ussd_session(session_id, session, redis_client=redis_client)
        return _ussd_continue(
            "Karibu Kipaji!\n"
            "Tuambie biashara yako ya leo.\n"
            "Mfano: Niliwauzia wateja mchele 3kg kwa 150 kila moja\n\n"
            "Andika ujumbe wako: "
        )

    if step == 1:
        trade_message = text.strip()
        if len(trade_message) < 5:
            return _ussd_end("Ujumbe mfupi sana. Tafadhali jaribu tena. *384#")

        user_lang = detect_language(trade_message)
        merchant_data = await get_merchant_data(merchant_id, db=db)
        velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

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

        if decision.confidence_gate_triggered:
            session["step"] = 2
            session["partial_message"] = trade_message
            await save_ussd_session(session_id, session, redis_client=redis_client)
            clarification = decision.response_message[:160]
            return _ussd_continue(clarification + "\n\nJibu: ")

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
            db=db,
        )
        background_tasks.add_task(
            _emit_audit, "ussd_credit_decision", merchant_id,
            {"tier": decision.credit_tier, "approved_local": decision.approved_local},
        )

        await clear_ussd_session(session_id, redis_client=redis_client)
        result_msg = decision.response_message[:160]
        return _ussd_end(result_msg)

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
            decision.model_dump(), sanitized.bias_proxy_removed, user_lang, db,
        )

        await clear_ussd_session(session_id, redis_client=redis_client)
        return _ussd_end(decision.response_message[:160])

    await clear_ussd_session(session_id, redis_client=redis_client)
    return _ussd_end("Kuna tatizo. Tafadhali piga *384# tena.")

# CRITICAL FIX: Return PlainTextResponse to prevent JSON quote wrapping
def _ussd_continue(text: str) -> PlainTextResponse:
    """CON prefix tells the USSD gateway to keep the session open."""
    return PlainTextResponse(f"CON {text}")

def _ussd_end(text: str) -> PlainTextResponse:
    """END prefix tells the USSD gateway to close the session."""
    return PlainTextResponse(f"END {text}")

# ---------------------------------------------------------------------------
# UTILITY ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "service": "Kipaji Core AI",
        "version": "4.0.0",
        "status": "ONLINE",
        "gemini": "connected" if ai_client else "unavailable (rule-based fallback active)",
        "firestore": "connected" if db else "unavailable (in-memory fallback active)",
        "redis": "connected" if redis_client else "unavailable (in-memory sessions active)",
    }

# CRITICAL FIX: Accept both GET and HEAD requests for UptimeRobot
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Lightweight health check for uptime monitoring and judge demos."""
    checks = {
        "api": True,
        "gemini_configured": bool(GEMINI_API_KEY),
        "firestore": False,
        "redis": False,
    }
    if db:
        try:
            await asyncio.to_thread(lambda: db.collection("_health").document("ping").get())
            checks["firestore"] = True
        except Exception:
            pass
    if redis_client:
        try:
            await asyncio.to_thread(redis_client.ping)
            checks["redis"] = True
        except Exception:
            pass
    
    overall = checks["api"] and checks["gemini_configured"]
    return {"healthy": overall, "checks": checks, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v1/merchant/{merchant_id}/history")
async def get_merchant_history(merchant_id: str):
    """Retrieve a merchant's trade history and velocity metrics."""
    data = await get_merchant_data(merchant_id, db=db)
    metrics = calculate_velocity_metrics(data["history"])
    return {
        "merchant_id": merchant_id,
        "trade_history_count": len(data["history"]),
        "recent_history": data["history"][-10:],
        "velocity_metrics": metrics,
        "profile": data["profile"],
    }

# JUDGE AUDIT ENDPOINT
@app.get("/api/v1/decisions")
async def get_audit_log():
    """Returns the last 20 AI credit decisions for judge verification."""
    return {
        "total_decisions": len(telemetry_history),
        "recent_decisions": telemetry_history
    }

# ---------------------------------------------------------------------------
# LIVE DASHBOARD (HTML/JS)
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kipaji AI - Live Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f6; color: #333; margin: 0; padding: 0; }
        header { background: linear-gradient(135deg, #2c3e50, #4ca1af); color: white; padding: 40px 20px; text-align: center; }
        header h1 { margin: 0; font-size: 2.5em; }
        header p { font-size: 1.2em; opacity: 0.9; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .card { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }
        .card h2 { color: #2c3e50; border-bottom: 2px solid #4ca1af; padding-bottom: 10px; }
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .feature-box { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #4ca1af; }
        .feature-box h3 { margin-top: 0; color: #2c3e50; }
        #telemetry-feed { height: 400px; overflow-y: auto; background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em; }
        .log-entry { background: #34495e; margin-bottom: 10px; padding: 10px; border-radius: 4px; border-left: 4px solid #4ca1af; }
        .log-entry.declined { border-left-color: #e74c3c; }
        .log-entry.approved { border-left-color: #2ecc71; }
        .log-header { font-weight: bold; color: #f1c40f; margin-bottom: 5px; }
        .bias-tag { background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-right: 5px; }
        .audit-trail { font-size: 0.85em; color: #bdc3c7; margin-top: 5px; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-online { background: #2ecc71; box-shadow: 0 0 10px #2ecc71; }
        .status-offline { background: #e74c3c; }
    </style>
</head>
<body>
    <header>
        <h1>🌾 Kipaji AI</h1>
        <p>Cash-Velocity Credit Engine for the Informal Economy</p>
    </header>
    <div class="container">
        <div class="card">
            <h2>What is Kipaji AI?</h2>
            <p>Kipaji AI is an AI-native fintech platform designed to bridge the credit gap for informal Micro, Small, and Medium Enterprises (MSMEs) in East Africa. Traditional banking relies on static credit histories that exclude millions of hardworking merchants. Kipaji changes this by analyzing <strong>conversational cash-velocity data</strong> via WhatsApp, SMS, and USSD.</p>
            <div class="features">
                <div class="feature-box">
                    <h3>🗣️ Conversational Underwriting</h3>
                    <p>Merchants simply describe their daily trade in their local language (Swahili, Sheng, Dholuo). Our AI extracts revenue signals and issues micro-credit in seconds.</p>
                </div>
                <div class="feature-box">
                    <h3>🛡️ Active Bias Mitigation</h3>
                    <p>Our Agent 1 Guardrail actively strips demographic proxies (gender, ethnicity, location) <em>before</em> the credit scoring agent sees the data, ensuring fair, objective lending.</p>
                </div>
                <div class="feature-box">
                    <h3> USSD & WhatsApp Native</h3>
                    <p>Built for the reality of African telecom infrastructure. Low-latency AI inference ensures decisions are delivered before USSD sessions timeout.</p>
                </div>
            </div>
        </div>
        <div class="card">
            <h2>
                <span class="status-indicator status-online" id="ws-status"></span>
                Live Telemetry: Bias Mitigation & Credit Decisions
            </h2>
            <p>Watch in real-time as Kipaji processes trade messages, strips biased proxies, and makes credit decisions. This feed is powered by our fire-and-forget WebSocket audit hub.</p>
            <div id="telemetry-feed">
                <p style="text-align:center; color:#bdc3c7;">Waiting for live telemetry data... Dial *384*63466# or send a WhatsApp message to see the AI in action!</p>
            </div>
        </div>
    </div>
    <script>
        const feed = document.getElementById('telemetry-feed');
        const statusDot = document.getElementById('ws-status');
        const wsUrl = `wss://${window.location.hostname}/ws/telemetry`;
        let ws;
        function connectWs() {
            ws = new WebSocket(wsUrl);
            ws.onopen = () => { statusDot.className = 'status-indicator status-online'; };
            ws.onclose = () => { statusDot.className = 'status-indicator status-offline'; setTimeout(connectWs, 3000); };
            ws.onmessage = (event) => { renderLog(JSON.parse(event.data)); };
        }
        function renderLog(data) {
            if (feed.querySelector('p')) feed.innerHTML = '';
            const entry = document.createElement('div');
            const isApproved = data.approved_local > 0;
            entry.className = `log-entry ${isApproved ? 'approved' : 'declined'}`;
            const biasTags = (data.bias_proxy_removed && data.bias_proxy_removed.length > 0) 
                ? data.bias_proxy_removed.map(b => `<span class="bias-tag">Stripped: ${b}</span>`).join('') 
                : '<span style="color:#2ecc71;">✅ No bias proxies detected</span>';
            const auditTrail = (data.audit_trail && data.audit_trail.length > 0) ? data.audit_trail.join(' • ') : 'No audit trail';
            entry.innerHTML = `
                <div class="log-header">${isApproved ? '✅ APPROVED' : '❌ DECLINED'} | Merchant: ${data.merchant_id} | Tier: ${data.tier || data.credit_tier || 'N/A'} | Amount: KSH ${data.approved_local || 0}</div>
                <div><strong>Bias Mitigation:</strong> ${biasTags}</div>
                <div class="audit-trail"><strong>Audit Trail:</strong> ${auditTrail}</div>
                <div style="font-size:0.75em; color:#7f8c8d; margin-top:5px;">${data.timestamp}</div>
            `;
            feed.prepend(entry);
            while (feed.children.length > 20) feed.removeChild(feed.lastChild);
        }
        connectWs();
    </script>
</body>
</html>
"""

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Live dashboard with real-time telemetry visualization."""
    return DASHBOARD_HTML

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