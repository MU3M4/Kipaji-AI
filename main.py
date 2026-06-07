"""
main.py — Kipaji Core AI v3.2 (Dashboard & Stateless USSD Edition)
"""
import os
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

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

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("KipajiCoreAI")

# ---------------------------------------------------------------------------
# CONFIGURATION & CLIENTS
# ---------------------------------------------------------------------------
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS    = os.getenv("ALLOWED_ORIGINS", "*").split(",")

ai_client = None
try:
    if GROQ_API_KEY:
        from openai import OpenAI
        ai_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        logger.info("Groq AI client initialised")
except Exception as e:
    logger.error(f"Groq client init failed: {e}")

db = None # Force in-memory fallback for hackathon stability

redis_client = None
try:
    import redis as redis_lib
    redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
    redis_client.ping()
    logger.info("Redis initialised")
except Exception as e:
    logger.warning(f"Redis unavailable — in-memory USSD sessions active: {e}")

# ---------------------------------------------------------------------------
# APP & MIDDLEWARE
# ---------------------------------------------------------------------------
app = FastAPI(title="Kipaji Core AI", version="3.2.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# TELEMETRY HUB (WebSocket)
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

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections: return
        payload = json.dumps(message, default=str)
        dead = []
        for conn in self.active_connections[:]:
            try: await conn.send_text(payload)
            except Exception: dead.append(conn)
        for ws in dead: self.disconnect(ws)

telemetry_hub = TelemetryHub()

async def _emit_audit(event_type: str, merchant_id: str, data: Dict[str, Any]):
    try:
        await telemetry_hub.broadcast({
            "event": event_type, "merchant_id": merchant_id,
            "timestamp": datetime.utcnow().isoformat(), **data,
        })
    except Exception as e:
        logger.debug(f"Telemetry emit error: {e}")

# ---------------------------------------------------------------------------
# DATA MODELS & HELPERS
# ---------------------------------------------------------------------------
class GatewayMessage(BaseModel):
    merchant_id: str = Field(..., min_length=3, max_length=64)
    phone_number: str = Field(..., pattern=r"^\+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1, max_length=2000)

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
    except Exception: return "sw"

def calculate_velocity_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    sales = [t for t in history if t.get("type") in ("revenue", "sale_entry_parsed") and isinstance(t.get("amount_local"), (int, float)) and t.get("amount_local", 0) > 0]
    if not sales: return {"avg_daily": 0.0, "total_7d": 0.0, "consistency": 1.0, "transaction_count": 0}
    recent = sales[-7:]
    amounts = [float(t["amount_local"]) for t in recent]
    total_7d = sum(amounts)
    avg_daily = total_7d / 7.0
    if len(amounts) > 1:
        mean = total_7d / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std = variance ** 0.5
        consistency = max(0.0, 1.0 - (std / (mean + 1)))
    else: consistency = 0.5
    return {"avg_daily": round(avg_daily, 2), "total_7d": round(total_7d, 2), "consistency": round(consistency, 2), "transaction_count": len(sales)}

def _extract_primary_trade_amount(sanitized: SanitizedInput) -> float:
    revenue_events = [e for e in sanitized.detected_trade_events if e.event_type in ("revenue", "receivable") and e.amount_ksh is not None and e.confidence >= 0.5]
    if not revenue_events: return 0.0
    return max(revenue_events, key=lambda e: e.confidence).amount_ksh

# ---------------------------------------------------------------------------
# WEBSOCKET ENDPOINT
# ---------------------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_hub.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: telemetry_hub.disconnect(websocket)

# ---------------------------------------------------------------------------
# WHATSAPP / API GATEWAY
# ---------------------------------------------------------------------------
@app.post("/api/v1/gateway")
async def inbound_telecom_gateway(payload: GatewayMessage, background_tasks: BackgroundTasks):
    merchant_id = payload.merchant_id
    user_lang = detect_language(payload.message_body)
    merchant_data = await get_merchant_data(merchant_id, db=db)
    velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

    sanitized = await run_bias_mitigation_guardrail(payload.message_body, merchant_id, user_lang, ai_client)
    decision = await run_kipaji_underwriter_core(sanitized, merchant_data["history"], merchant_data["profile"], velocity_metrics, merchant_id, ai_client)

    extracted_amount = _extract_primary_trade_amount(sanitized)
    background_tasks.add_task(save_trade_interaction, merchant_id, extracted_amount, (sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"), decision.model_dump(), sanitized.bias_proxy_removed, user_lang, db)
    background_tasks.add_task(_emit_audit, "credit_decision", merchant_id, {"tier": decision.credit_tier, "approved_local": decision.approved_local, "bias_proxy_removed": sanitized.bias_proxy_removed, "audit_trail": decision.audit_trail})

    return {"status": "SUCCESS", "merchant_id": merchant_id, "language_detected": user_lang, "credit_decision": decision.model_dump(), "velocity_metrics": velocity_metrics, "response_message": decision.response_message}

# ---------------------------------------------------------------------------
# STATELESS USSD GATEWAY (Africa's Talking)
# ---------------------------------------------------------------------------
@app.post("/api/v1/ussd", response_model=None)
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    session_id   = form.get("sessionId", "")
    phone_number = form.get("phoneNumber", "")
    text         = form.get("text", "")
    
    merchant_id = "m_" + phone_number.replace("+", "").replace(" ", "")
    logger.info(f"[USSD] session={session_id} phone={phone_number} text='{text}'")
    
    # Africa's Talking accumulates inputs separated by '*'
    parts = text.split('*') if text else []
    step = len(parts)
    
    # Step 0: Opt-in
    if step == 0:
        return _ussd_continue("Welcome to Kipaji AI!\nBy continuing you agree to our terms.\n1. Opt In\n2. Opt Out")
        
    # Step 1: Language Selection
    if step == 1:
        if parts[0] != '1':
            return _ussd_end("Thank you for trying Kipaji AI. Goodbye!")
        return _ussd_continue("Choose your language:\n1. English\n2. Swahili")
        
    # Step 2: Trade Input Prompt
    if step == 2:
        lang_choice = parts[1]
        if lang_choice == '1':
            return _ussd_continue("Enter your trade message for today:\nE.g. Sold maize 500 KSH")
        else:
            return _ussd_continue("Ingiza ujumbe wa biashara yako ya leo:\nMfano: Nimeuza mahindi 500")
            
    # Step 3+: Process Trade
    if step >= 3:
        lang_choice = parts[1]
        lang = "en" if lang_choice == '1' else "sw"
        trade_message = '*'.join(parts[2:]) # Rejoin in case they typed a '*'
        
        if len(trade_message.strip()) < 5:
            prompt = "Message too short. Please dial *384*63463466# and try again." if lang == 'en' else "Ujumbe mfupi sana. Tafadhali jaribu tena."
            return _ussd_end(prompt)
            
        merchant_data = await get_merchant_data(merchant_id, db=db)
        velocity_metrics = calculate_velocity_metrics(merchant_data["history"])
        
        sanitized = await run_bias_mitigation_guardrail(trade_message, merchant_id, lang, ai_client)
        decision = await run_kipaji_underwriter_core(sanitized, merchant_data["history"], merchant_data["profile"], velocity_metrics, merchant_id, ai_client)
        
        extracted_amount = _extract_primary_trade_amount(sanitized)
        background_tasks.add_task(save_trade_interaction, merchant_id, extracted_amount, (sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"), decision.model_dump(), sanitized.bias_proxy_removed, lang, db)
        
        # Emit to WebSocket Dashboard!
        background_tasks.add_task(_emit_audit, "ussd_credit_decision", merchant_id, {
            "tier": decision.credit_tier, 
            "approved_local": decision.approved_local, 
            "bias_proxy_removed": sanitized.bias_proxy_removed, 
            "audit_trail": decision.audit_trail
        })
        
        return _ussd_end(decision.response_message[:160])
        
    return _ussd_end("Session error. Please dial *384*63466# again.")

def _ussd_continue(text: str) -> str: return f"CON {text}"
def _ussd_end(text: str) -> str: return f"END {text}"

# ---------------------------------------------------------------------------
# LIVE DASHBOARD (Replaces the old JSON root endpoint)
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
                    <h3>⚡ USSD & WhatsApp Native</h3>
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return DASHBOARD_HTML

@app.get("/health")
async def health_check():
    return {"healthy": True, "checks": {"api": True, "groq_configured": bool(GROQ_API_KEY)}, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v1/merchant/{merchant_id}/history")
async def get_merchant_history(merchant_id: str):
    data = await get_merchant_data(merchant_id, db=db)
    metrics = calculate_velocity_metrics(data["history"])
    return {"merchant_id": merchant_id, "trade_history_count": len(data["history"]), "recent_history": data["history"][-10:], "velocity_metrics": metrics, "profile": data["profile"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=os.getenv("ENV", "production") == "development", log_level="info")