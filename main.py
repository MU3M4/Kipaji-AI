"""
main.py — Kipaji Core AI v4.0 (Hybrid AI & Live Dashboard Edition)
"""
import os, json, logging, asyncio
from typing import Any, Dict, List
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from agents import run_bias_mitigation_guardrail, run_kipaji_underwriter_core, SanitizedInput
from storage import get_merchant_data, save_trade_interaction
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("KipajiCoreAI")

# --- HYBRID AI CLIENTS ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

gemini_client = None
groq_client = None

try:
    if GEMINI_API_KEY:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI client initialised (Agent 1: Bias Guardrail)")
except Exception as e: logger.error(f"Gemini client init failed: {e}")

try:
    if GROQ_API_KEY:
        from openai import OpenAI
        groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        logger.info("Groq AI client initialised (Agent 2: Underwriter)")
except Exception as e: logger.error(f"Groq client init failed: {e}")

app = FastAPI(title="Kipaji Core AI", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- TELEMETRY HUB (WITH HISTORY CACHE) ---
telemetry_history = []
class TelemetryHub:
    def __init__(self): self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
    async def broadcast(self, message: Dict[str, Any]):
        telemetry_history.append(message)
        if len(telemetry_history) > 20: telemetry_history.pop(0)
        if not self.active_connections: return
        payload = json.dumps(message, default=str)
        for conn in self.active_connections[:]:
            try: await conn.send_text(payload)
            except Exception: self.disconnect(conn)

telemetry_hub = TelemetryHub()

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_hub.connect(websocket)
    try:
        for msg in telemetry_history: await websocket.send_text(json.dumps(msg, default=str))
        while True: await websocket.receive_text()
    except WebSocketDisconnect: telemetry_hub.disconnect(websocket)

async def _emit_audit(event_type: str, merchant_id: str, data: Dict[str, Any]):
    try: 
        msg = {"event": event_type, "merchant_id": merchant_id, "timestamp": datetime.utcnow().isoformat(), **data}
        logger.info(f"[Telemetry] Broadcasting: {msg}")
        await telemetry_hub.broadcast(msg)
    except Exception as e: logger.error(f"[Telemetry] Broadcast failed: {e}")

# --- HELPERS ---
class GatewayMessage(BaseModel):
    merchant_id: str = Field(..., min_length=3)
    phone_number: str = Field(..., pattern=r"^\+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1)

_SWAHILI_MARKERS = {"nimeuza", "niliuza", "nilinunua", "biashara", "bei", "leo", "shilingi", "pesa", "ksh"}
def detect_language(text: str) -> str:
    tokens = set(text.lower().split())
    if tokens & _SWAHILI_MARKERS: return "sw"
    try: return "sw" if detect(text[:400]) == "sw" else "en"
    except Exception: return "en"

def calculate_velocity_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    sales = [t for t in history if t.get("type") == "revenue" and isinstance(t.get("amount_local"), (int, float)) and t.get("amount_local", 0) > 0]
    if not sales: return {"avg_daily": 0.0, "total_7d": 0.0, "consistency": 1.0, "transaction_count": 0}
    recent = sales[-7:]
    amounts = [float(t["amount_local"]) for t in recent]
    total_7d = sum(amounts)
    avg_daily = total_7d / 7.0
    consistency = 0.5
    if len(amounts) > 1:
        mean = total_7d / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        consistency = max(0.0, 1.0 - (variance ** 0.5 / (mean + 1)))
    return {"avg_daily": round(avg_daily, 2), "total_7d": round(total_7d, 2), "consistency": round(consistency, 2), "transaction_count": len(sales)}

def _extract_primary_trade_amount(sanitized: SanitizedInput) -> float:
    revenue_events = [e for e in sanitized.detected_trade_events if e.event_type in ("revenue", "receivable") and e.amount_ksh is not None and e.confidence >= 0.5]
    if not revenue_events: return 0.0
    return max(revenue_events, key=lambda e: e.confidence).amount_ksh

# --- GATEWAYS ---
@app.post("/api/v1/gateway")
async def inbound_telecom_gateway(payload: GatewayMessage, background_tasks: BackgroundTasks):
    merchant_id = payload.merchant_id
    user_lang = detect_language(payload.message_body)
    merchant_data = await get_merchant_data(merchant_id)
    velocity_metrics = calculate_velocity_metrics(merchant_data["history"])

    sanitized = await run_bias_mitigation_guardrail(payload.message_body, merchant_id, user_lang, gemini_client)
    decision = await run_kipaji_underwriter_core(sanitized, merchant_data["history"], merchant_data["profile"], velocity_metrics, merchant_id, groq_client)

    background_tasks.add_task(save_trade_interaction, merchant_id, _extract_primary_trade_amount(sanitized), (sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"), decision.model_dump(), sanitized.bias_proxy_removed, user_lang)
    background_tasks.add_task(_emit_audit, "credit_decision", merchant_id, {"tier": decision.credit_tier, "approved_local": decision.approved_local, "bias_proxy_removed": sanitized.bias_proxy_removed, "audit_trail": decision.audit_trail})
    return {"status": "SUCCESS", "merchant_id": merchant_id, "language_detected": user_lang, "credit_decision": decision.model_dump(), "velocity_metrics": velocity_metrics, "response_message": decision.response_message}

@app.post("/api/v1/ussd")
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    phone_number = form.get("phoneNumber", "")
    text = form.get("text", "")
    merchant_id = "m_" + phone_number.replace("+", "").replace(" ", "")
    parts = text.split('*') if text else []
    step = len(parts)
    
    if step == 0: return PlainTextResponse("CON Welcome to Kipaji AI!\nBy continuing you agree to our terms.\n1. Opt In\n2. Opt Out")
    if step == 1:
        if parts[0] != '1': return PlainTextResponse("END Thank you for trying Kipaji AI. Goodbye!")
        return PlainTextResponse("CON Choose your language:\n1. English\n2. Swahili")
    if step == 2:
        if parts[1] == '1': return PlainTextResponse("CON Enter your trade message for today:\nE.g. Sold maize 500 KSH")
        else: return PlainTextResponse("CON Ingiza ujumbe wa biashara yako ya leo:\nMfano: Nimeuza mahindi 500")
    if step >= 3:
        lang = "en" if parts[1] == '1' else "sw"
        trade_message = '*'.join(parts[2:])
        if len(trade_message.strip()) < 5:
            prompt = "Message too short. Please dial *384*63466# and try again." if lang == 'en' else "Ujumbe mfupi sana. Tafadhali jaribu tena."
            return PlainTextResponse(f"END {prompt}")
            
        merchant_data = await get_merchant_data(merchant_id)
        velocity_metrics = calculate_velocity_metrics(merchant_data["history"])
        sanitized = await run_bias_mitigation_guardrail(trade_message, merchant_id, lang, gemini_client)
        decision = await run_kipaji_underwriter_core(sanitized, merchant_data["history"], merchant_data["profile"], velocity_metrics, merchant_id, groq_client)
        
        background_tasks.add_task(save_trade_interaction, merchant_id, _extract_primary_trade_amount(sanitized), (sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"), decision.model_dump(), sanitized.bias_proxy_removed, lang)
        background_tasks.add_task(_emit_audit, "ussd_credit_decision", merchant_id, {"tier": decision.credit_tier, "approved_local": decision.approved_local, "bias_proxy_removed": sanitized.bias_proxy_removed, "audit_trail": decision.audit_trail})
        return PlainTextResponse(f"END {decision.response_message[:160]}")
    return PlainTextResponse("END Session error. Please dial *384*63466# again.")

# --- AUDIT ENDPOINT FOR JUDGES ---
@app.get("/api/v1/decisions")
async def get_audit_log():
    return {"total_decisions": len(telemetry_history), "recent_decisions": telemetry_history}

# --- LIVE DASHBOARD ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kipaji AI - Live Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f4f7f6; color: #333; margin: 0; padding: 0; }
        header { background: linear-gradient(135deg, #2c3e50, #4ca1af); color: white; padding: 40px 20px; text-align: center; }
        header h1 { margin: 0; font-size: 2.5em; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .card { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }
        .card h2 { color: #2c3e50; border-bottom: 2px solid #4ca1af; padding-bottom: 10px; }
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .feature-box { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #4ca1af; }
        #telemetry-feed { height: 400px; overflow-y: auto; background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em; }
        .log-entry { background: #34495e; margin-bottom: 10px; padding: 10px; border-radius: 4px; border-left: 4px solid #4ca1af; }
        .log-entry.approved { border-left-color: #2ecc71; }
        .log-entry.declined { border-left-color: #e74c3c; }
        .log-header { font-weight: bold; color: #f1c40f; margin-bottom: 5px; }
        .bias-tag { background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-right: 5px; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-online { background: #2ecc71; box-shadow: 0 0 10px #2ecc71; }
    </style>
</head>
<body>
    <header><h1>🌾 Kipaji AI</h1><p>Hybrid AI Cash-Velocity Credit Engine for the Informal Economy</p></header>
    <div class="container">
        <div class="card">
            <h2>What is Kipaji AI?</h2>
            <p>Kipaji AI bridges the credit gap for informal MSMEs in East Africa by analyzing conversational cash-velocity data via WhatsApp, SMS, and USSD.</p>
            <div class="features">
                <div class="feature-box"><h3>🗣️ Conversational Underwriting</h3><p>Merchants describe their daily trade in Swahili or English. Our AI extracts revenue signals and issues micro-credit in seconds.</p></div>
                <div class="feature-box"><h3>🛡️ Active Bias Mitigation</h3><p>Agent 1 (Gemini) strips demographic proxies (gender, ethnicity, location) BEFORE the credit scoring agent sees the data.</p></div>
                <div class="feature-box"><h3>⚡ Hybrid AI Architecture</h3><p>We use Gemini for nuanced cultural bias-mitigation, and Groq (Llama 3.3) for sub-second underwriting to ensure USSD sessions never timeout.</p></div>
            </div>
        </div>
        <div class="card">
            <h2><span class="status-indicator status-online"></span>Live Telemetry: Bias Mitigation & Credit Decisions</h2>
            <p>Watch in real-time as Kipaji processes trade messages, strips biased proxies, and makes credit decisions. Dial <strong>*384*63466#</strong> to see the AI in action!</p>
            <div id="telemetry-feed"><p style="text-align:center; color:#bdc3c7;">Waiting for live telemetry data...</p></div>
        </div>
    </div>
    <script>
        const feed = document.getElementById('telemetry-feed');
        const wsUrl = `wss://${window.location.hostname}/ws/telemetry`;
        let ws;
        function connectWs() {
            ws = new WebSocket(wsUrl);
            ws.onmessage = (event) => { renderLog(JSON.parse(event.data)); };
            ws.onclose = () => { setTimeout(connectWs, 3000); };
        }
        function renderLog(data) {
            if (feed.querySelector('p')) feed.innerHTML = '';
            const entry = document.createElement('div');
            const isApproved = data.approved_local > 0;
            entry.className = `log-entry ${isApproved ? 'approved' : 'declined'}`;
            const biasTags = (data.bias_proxy_removed && data.bias_proxy_removed.length > 0) 
                ? data.bias_proxy_removed.map(b => `<span class="bias-tag">Stripped: ${b}</span>`).join('') 
                : '<span style="color:#2ecc71;">✅ No bias proxies detected</span>';
            entry.innerHTML = `
                <div class="log-header">${isApproved ? '✅ APPROVED' : '❌ DECLINED'} | Merchant: ${data.merchant_id} | Tier: ${data.tier || 'N/A'} | Amount: KSH ${data.approved_local || 0}</div>
                <div><strong>Bias Mitigation:</strong> ${biasTags}</div>
                <div style="font-size:0.85em; color:#bdc3c7; margin-top:5px;"><strong>Audit Trail:</strong> ${(data.audit_trail || []).join(' • ')}</div>
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
async def read_root(): return DASHBOARD_HTML

@app.get("/health")
async def health_check():
    return {"healthy": True, "checks": {"api": True, "gemini_configured": bool(gemini_client), "groq_configured": bool(groq_client)}, "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")