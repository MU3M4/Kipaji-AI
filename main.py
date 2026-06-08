import os
import json
import logging
import asyncio
from typing import Any, Dict, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

load_dotenv()
from agents import run_bias_mitigation_guardrail, run_kipaji_underwriter_core, SanitizedInput
from storage import get_merchant_data, save_trade_interaction, get_ussd_session, save_ussd_session, clear_ussd_session

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("KipajiCoreAI")

# --- HYBRID CLIENTS ---
gemini_client = None
groq_client = None
try:
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("Gemini AI client initialised")
except Exception as e: logger.error(f"Gemini init failed: {e}")

try:
    if os.getenv("GROQ_API_KEY"):
        from openai import OpenAI
        groq_client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        logger.info("Groq AI client initialised")
except Exception as e: logger.error(f"Groq init failed: {e}")

db = None
redis_client = None

app = FastAPI(title="Kipaji Core AI", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- TELEMETRY HUB ---
telemetry_history = []
class TelemetryHub:
    def __init__(self): self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        for msg in telemetry_history: await websocket.send_text(json.dumps(msg, default=str))
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
        while True: await websocket.receive_text()
    except WebSocketDisconnect: telemetry_hub.disconnect(websocket)

async def _emit_audit(event_type: str, merchant_id: str, data: Dict[str, Any]):
    try: await telemetry_hub.broadcast({"event": event_type, "merchant_id": merchant_id, "timestamp": datetime.utcnow().isoformat(), **data})
    except Exception: pass

# --- HELPERS ---
class GatewayMessage(BaseModel):
    merchant_id: str = Field(..., min_length=3)
    phone_number: str = Field(..., pattern=r"^\+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1)

_SWAHILI_MARKERS = {"nimeuza", "niliuza", "biashara", "bei", "leo", "shilingi", "pesa", "ksh"}
def detect_language(text: str) -> str:
    if set(text.lower().split()) & _SWAHILI_MARKERS: return "sw"
    try: return "sw" if detect(text[:400]) == "sw" else "en"
    except: return "en"

def calculate_velocity_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    sales = [t for t in history if t.get("type") == "revenue" and t.get("amount_local", 0) > 0]
    if not sales: return {"avg_daily": 0.0, "total_7d": 0.0, "consistency": 1.0, "transaction_count": 0}
    amounts = [float(t["amount_local"]) for t in sales[-7:]]
    return {"avg_daily": sum(amounts)/7, "total_7d": sum(amounts), "consistency": 0.5, "transaction_count": len(sales)}

def _extract_primary_trade_amount(sanitized: SanitizedInput) -> float:
    rev = [e for e in sanitized.detected_trade_events if e.event_type in ("revenue", "receivable") and e.amount_ksh and e.confidence >= 0.5]
    return max(rev, key=lambda e: e.confidence).amount_ksh if rev else 0.0

# --- GATEWAYS ---
@app.post("/api/v1/gateway")
async def inbound_telecom_gateway(payload: GatewayMessage, background_tasks: BackgroundTasks):
    user_lang = detect_language(payload.message_body)
    merchant_data = await get_merchant_data(payload.merchant_id, db=db)
    sanitized = await run_bias_mitigation_guardrail(payload.message_body, payload.merchant_id, user_lang, gemini_client)
    decision = await run_kipaji_underwriter_core(sanitized, merchant_data["history"], merchant_data["profile"], calculate_velocity_metrics(merchant_data["history"]), payload.merchant_id, groq_client)
    background_tasks.add_task(save_trade_interaction, payload.merchant_id, _extract_primary_trade_amount(sanitized), "revenue", decision.model_dump(), sanitized.bias_proxy_removed, user_lang, db)
    background_tasks.add_task(_emit_audit, "credit_decision", payload.merchant_id, {"tier": decision.credit_tier, "approved_local": decision.approved_local, "bias_proxy_removed": sanitized.bias_proxy_removed, "audit_trail": decision.audit_trail})
    return {"status": "SUCCESS", "credit_decision": decision.model_dump()}

@app.post("/api/v1/ussd")
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    phone, text = form.get("phoneNumber", ""), form.get("text", "")
    merchant_id = "m_" + phone.replace("+", "").replace(" ", "")
    parts = text.split('*') if text else []
    step = len(parts)
    
    if step == 0: return PlainTextResponse("CON Karibu Kipaji!\n1. Opt In\n2. Opt Out")
    if step == 1:
        if parts[0] != '1': return PlainTextResponse("END Goodbye!")
        return PlainTextResponse("CON Language:\n1. English\n2. Swahili")
    if step == 2: return PlainTextResponse("CON Enter trade message:\nE.g. Nimeuza mahindi 1500")
    if step >= 3:
        lang = "en" if parts[1] == '1' else "sw"
        trade = '*'.join(parts[2:])
        merchant_data = await get_merchant_data(merchant_id, db=db)
        sanitized = await run_bias_mitigation_guardrail(trade, merchant_id, lang, gemini_client)
        decision = await run_kipaji_underwriter_core(sanitized, merchant_data["history"], merchant_data["profile"], calculate_velocity_metrics(merchant_data["history"]), merchant_id, groq_client)
        background_tasks.add_task(save_trade_interaction, merchant_id, _extract_primary_trade_amount(sanitized), "revenue", decision.model_dump(), sanitized.bias_proxy_removed, lang, db)
        background_tasks.add_task(_emit_audit, "ussd_decision", merchant_id, {"tier": decision.credit_tier, "approved_local": decision.approved_local, "bias_proxy_removed": sanitized.bias_proxy_removed, "audit_trail": decision.audit_trail})
        return PlainTextResponse(f"END {decision.response_message[:160]}")
    return PlainTextResponse("END Error")

# --- ENDPOINTS ---
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check(): return {"healthy": True}

@app.get("/api/v1/decisions")
async def get_audit_log(): return {"total": len(telemetry_history), "decisions": telemetry_history}

DASHBOARD_HTML = """<!DOCTYPE html><html><head><title>Kipaji AI</title><style>body{font-family:sans-serif;background:#f4f7f6;text-align:center;padding:40px;} .card{background:#2c3e50;color:#fff;padding:20px;border-radius:8px;max-width:800px;margin:0 auto;} .approved{border-left:10px solid #2ecc71;} .log{background:#34495e;padding:10px;margin:10px 0;text-align:left;border-radius:4px;}</style></head><body><h1>🌾 Kipaji AI Live Telemetry</h1><div id="feed" class="card">Waiting for USSD...</div><script>const f=document.getElementById('feed');const ws=new WebSocket(`wss://${window.location.hostname}/ws/telemetry`);ws.onmessage=e=>{const d=JSON.parse(e.data);const el=document.createElement('div');el.className='log approved';el.innerHTML=`<b>✅ APPROVED</b> | ${d.merchant_id} | KSH ${d.approved_local}<br>Stripped: ${(d.bias_proxy_removed||[]).join(', ')||'None'}`;f.prepend(el);};</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
async def read_root(): return DASHBOARD_HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")