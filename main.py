"""
main.py — Kipaji Core AI v4.0 (Production-Ready Edition)
"""
import os, json, logging, asyncio, base64, hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

load_dotenv()
from agents import run_bias_mitigation_guardrail, run_kipaji_underwriter_core, SanitizedInput, CreditDecision
from storage import get_merchant_data, save_trade_interaction, get_ussd_session, save_ussd_session, clear_ussd_session

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("KipajiCoreAI")

# CONFIGURATION
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
MPESA_CONSUMER_KEY = os.getenv("MPESA_CONSUMER_KEY")
MPESA_CONSUMER_SECRET = os.getenv("MPESA_CONSUMER_SECRET")
MPESA_SHORTCODE = os.getenv("MPESA_SHORTCODE", "174379")
MPESA_PASSKEY = os.getenv("MPESA_PASSKEY")
MPESA_CALLBACK_URL = os.getenv("MPESA_CALLBACK_URL", "https://kipaji-ai.onrender.com/api/v1/mpesa/callback")
MPESA_ENV = os.getenv("MPESA_ENV", "sandbox")
MPESA_BASE_URL = "https://sandbox.safaricom.co.ke" if MPESA_ENV == "sandbox" else "https://api.safaricom.co.ke"

# STATE
_PAYMENT_LEDGER: Dict[str, Dict] = {}
_DECISION_LOG: List[Dict[str, Any]] = []
_DECISION_LOG_MAX = 500

# CLIENTS
ai_client = None
try:
    if GEMINI_API_KEY:
        from google import genai
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI client initialised")
except Exception as e: logger.error(f"Gemini init failed: {e}")

db = None
try:
    from google.cloud import firestore
    db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
    logger.info("Firestore initialised")
except Exception as e: logger.warning(f"Firestore unavailable: {e}")

redis_client = None
try:
    import redis as redis_lib
    redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
    redis_client.ping()
    logger.info("Redis initialised")
except Exception as e: logger.warning(f"Redis unavailable: {e}")

app = FastAPI(title="Kipaji Core AI", version="4.0.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["GET", "POST", "HEAD", "OPTIONS"], allow_headers=["*"])

# TELEMETRY HUB
telemetry_history = []
MAX_HISTORY = 20
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
        if len(telemetry_history) > MAX_HISTORY: telemetry_history.pop(0)
        if not self.active_connections: return
        payload = json.dumps(message, default=str)
        dead = [conn for conn in self.active_connections[:] if (await conn.send_text(payload)) is None or False] # Simplified error handling
        # Better error handling for broadcast
        for conn in self.active_connections[:]:
            try: await conn.send_text(payload)
            except: self.disconnect(conn)

telemetry_hub = TelemetryHub()

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_hub.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: telemetry_hub.disconnect(websocket)

async def _emit_audit(event_type: str, merchant_id: str, data: Dict[str, Any]):
    try: await telemetry_hub.broadcast({"event": event_type, "merchant_id": merchant_id, "timestamp": datetime.utcnow().isoformat(), **data})
    except: pass

# AUDIT LOG
def log_credit_decision(merchant_id, channel, language, sanitized, decision, velocity_metrics):
    timestamp = datetime.utcnow().isoformat()
    decision_id = hashlib.sha256(f"{merchant_id}{timestamp}".encode()).hexdigest()[:8]
    entry = {
        "decision_id": decision_id, "timestamp": timestamp, "merchant_id": merchant_id, "channel": channel, "language_detected": language,
        "agent_1_output": {"overall_confidence": sanitized.overall_confidence, "trade_events_extracted": len(sanitized.detected_trade_events), "bias_proxies_removed": sanitized.bias_proxy_removed, "low_confidence_events": sanitized.low_confidence_count},
        "confidence_gate_triggered": decision.confidence_gate_triggered,
        "agent_2_output": {"credit_tier": decision.credit_tier, "approved_local_ksh": decision.approved_local, "velocity_score": decision.velocity_score, "consistency_score": decision.consistency_score, "interest_rate_monthly_pct": decision.interest_rate_monthly, "repayment_days": decision.repayment_days},
        "velocity_metrics_at_decision": velocity_metrics, "decision_reason": decision.decision_reason, "audit_trail": decision.audit_trail, "approved": decision.approved
    }
    _DECISION_LOG.append(entry)
    if len(_DECISION_LOG) > _DECISION_LOG_MAX: _DECISION_LOG.pop(0)

# HELPERS
class GatewayMessage(BaseModel):
    merchant_id: str = Field(..., min_length=3, max_length=64)
    phone_number: str = Field(..., pattern=r"^\+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1, max_length=2000)

_SWAHILI_MARKERS = {"nimeuza", "niliuza", "biashara", "bei", "leo", "shilingi", "pesa", "ksh"}
_SHENG_MARKERS = {"niko na", "nimefanya", "chapaa", "fiti"}
def detect_language(text: str) -> str:
    tokens = set(text.lower().split())
    if tokens & _SWAHILI_MARKERS or tokens & _SHENG_MARKERS: return "sw"
    try: return "sw" if detect(text[:400]) == "sw" else "en"
    except: return "en"

def calculate_velocity_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    sales = [t for t in history if t.get("type") in ("revenue", "sale_entry_parsed") and isinstance(t.get("amount_local"), (int, float)) and t.get("amount_local", 0) > 0]
    if not sales: return {"avg_daily": 0.0, "total_7d": 0.0, "consistency": 1.0, "transaction_count": 0}
    amounts = [float(t["amount_local"]) for t in sales[-7:]]
    total_7d = sum(amounts)
    avg_daily = total_7d / 7.0
    consistency = 0.5
    if len(amounts) > 1:
        mean = total_7d / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        consistency = max(0.0, 1.0 - ((variance ** 0.5) / (mean + 1)))
    return {"avg_daily": round(avg_daily, 2), "total_7d": round(total_7d, 2), "consistency": round(consistency, 2), "transaction_count": len(sales)}

def _extract_primary_trade_amount(sanitized: SanitizedInput) -> float:
    rev = [e for e in sanitized.detected_trade_events if e.event_type in ("revenue", "receivable") and e.amount_ksh and e.confidence >= 0.5]
    return max(rev, key=lambda e: e.confidence).amount_ksh if rev else 0.0

# M-PESA
async def get_mpesa_access_token() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{MPESA_BASE_URL}/oauth/v1/generate?grant_type=client_credentials", auth=(MPESA_CONSUMER_KEY, MPESA_CONSUMER_SECRET))
        resp.raise_for_status()
        return resp.json()["access_token"]

async def trigger_mpesa_stk_push(phone_number: str, merchant_id: str, amount: int = 50) -> dict:
    try:
        token = await get_mpesa_access_token()
        phone = phone_number.replace("+", "").replace(" ", "").replace("-", "")
        if not phone.startswith("254"): phone = "254" + phone[1:] if phone.startswith("0") else "254" + phone
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        pwd = base64.b64encode(f"{MPESA_SHORTCODE}{MPESA_PASSKEY}{ts}".encode()).decode()
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{MPESA_BASE_URL}/mpesa/stkpush/v1/processrequest", json={"BusinessShortCode": MPESA_SHORTCODE, "Password": pwd, "Timestamp": ts, "TransactionType": "CustomerPayBillOnline", "Amount": amount, "PartyA": phone, "PartyB": MPESA_SHORTCODE, "PhoneNumber": phone, "CallBackURL": MPESA_CALLBACK_URL, "AccountReference": f"KIPAJI-{merchant_id[:8].upper()}", "TransactionDesc": "Kipaji Fee"}, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            return resp.json()
    except Exception as e:
        logger.error(f"M-Pesa STK Push failed: {e}")
        return {"error": str(e), "triggered": False}

# ENDPOINTS
@app.post("/api/v1/gateway")
async def inbound_telecom_gateway(payload: GatewayMessage, background_tasks: BackgroundTasks):
    mid = payload.merchant_id
    lang = detect_language(payload.message_body)
    mdata = await get_merchant_data(mid, db=db)
    vm = calculate_velocity_metrics(mdata["history"])
    san = await run_bias_mitigation_guardrail(payload.message_body, mid, lang, ai_client)
    dec = await run_kipaji_underwriter_core(san, mdata["history"], mdata["profile"], vm, mid, ai_client)
    
    bg_tasks = [
        save_trade_interaction(mid, _extract_primary_trade_amount(san), (san.detected_trade_events[0].event_type if san.detected_trade_events else "unknown"), dec.model_dump(), san.bias_proxy_removed, lang, db),
        _emit_audit("credit_decision", mid, {"channel": payload.channel, "language": lang, "credit_tier": dec.credit_tier, "approved_local": dec.approved_local, "bias_proxy_removed": san.bias_proxy_removed, "audit_trail": dec.audit_trail}),
        log_credit_decision(mid, payload.channel, lang, san, dec, vm)
    ]
    if dec.approved and MPESA_CONSUMER_KEY and MPESA_PASSKEY:
        bg_tasks.append(trigger_mpesa_stk_push(payload.phone_number, mid, 50))
        
    for task in bg_tasks: background_tasks.add_task(task)
    return {"status": "SUCCESS", "merchant_id": mid, "language_detected": lang, "credit_decision": dec.model_dump(), "velocity_metrics": vm, "response_message": dec.response_message, "fee_collection": "initiated" if dec.approved else "not_applicable"}

@app.post("/api/v1/ussd", response_model=None)
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    sid, ph, txt = form.get("sessionId", ""), form.get("phoneNumber", ""), form.get("text", "")
    mid = "m_" + ph.replace("+", "").replace(" ", "")
    sess = await get_ussd_session(sid, redis_client=redis_client) or {"step": 0, "merchant_id": mid, "phone": ph}
    step = sess.get("step", 0)
    
    if txt == "" or step == 0:
        sess["step"] = 1; await save_ussd_session(sid, sess, redis_client=redis_client)
        return PlainTextResponse("CON Karibu Kipaji!\nTuambie biashara yako ya leo.\nMfano: Niliwauzia wateja mchele 3kg kwa 150 kila moja\n\nAndika ujumbe wako: ")
    
    if step == 1:
        if len(txt.strip()) < 5: return PlainTextResponse("END Ujumbe mfupi sana. Jaribu tena.")
        lang = detect_language(txt)
        mdata = await get_merchant_data(mid, db=db)
        vm = calculate_velocity_metrics(mdata["history"])
        san = await run_bias_mitigation_guardrail(txt, mid, lang, ai_client)
        dec = await run_kipaji_underwriter_core(san, mdata["history"], mdata["profile"], vm, mid, ai_client)
        
        background_tasks.add_task(log_credit_decision, mid, "ussd", lang, san, dec, vm)
        
        if dec.confidence_gate_triggered:
            sess["step"] = 2; sess["partial_message"] = txt; await save_ussd_session(sid, sess, redis_client=redis_client)
            return PlainTextResponse(f"CON {dec.response_message[:160]}\n\nJibu: ")
            
        bg_tasks = [save_trade_interaction(mid, _extract_primary_trade_amount(san), (san.detected_trade_events[0].event_type if san.detected_trade_events else "unknown"), dec.model_dump(), san.bias_proxy_removed, lang, db), _emit_audit("ussd_credit_decision", mid, {"tier": dec.credit_tier, "approved_local": dec.approved_local})]
        if dec.approved and MPESA_CONSUMER_KEY and MPESA_PASSKEY: bg_tasks.append(trigger_mpesa_stk_push(ph, mid, 50))
        for t in bg_tasks: background_tasks.add_task(t)
        await clear_ussd_session(sid, redis_client=redis_client)
        return PlainTextResponse(f"END {dec.response_message[:160]}")
        
    if step == 2:
        combined = sess.get("partial_message", "") + " " + txt.strip()
        lang = detect_language(combined)
        mdata = await get_merchant_data(mid, db=db)
        vm = calculate_velocity_metrics(mdata["history"])
        san = await run_bias_mitigation_guardrail(combined, mid, lang, ai_client)
        dec = await run_kipaji_underwriter_core(san, mdata["history"], mdata["profile"], vm, mid, ai_client)
        background_tasks.add_task(save_trade_interaction, mid, _extract_primary_trade_amount(san), (san.detected_trade_events[0].event_type if san.detected_trade_events else "unknown"), dec.model_dump(), san.bias_proxy_removed, lang, db)
        background_tasks.add_task(log_credit_decision, mid, "ussd", lang, san, dec, vm)
        await clear_ussd_session(sid, redis_client=redis_client)
        return PlainTextResponse(f"END {dec.response_message[:160]}")
        
    await clear_ussd_session(sid, redis_client=redis_client)
    return PlainTextResponse("END Kuna tatizo. Piga *384# tena.")

@app.post("/api/v1/mpesa/callback")
async def mpesa_callback(request: Request):
    try:
        body = await request.json()
        stk = body.get("Body", {}).get("stkCallback", {})
        rc = stk.get("ResultCode")
        meta = {i["Name"]: i["Value"] for i in stk.get("CallbackMetadata", {}).get("Item", [])}
        receipt = meta.get("MpesaReceiptNumber")
        if receipt: _PAYMENT_LEDGER[receipt] = {"result_code": rc, "amount": meta.get("Amount"), "receipt": receipt, "timestamp": datetime.utcnow().isoformat()}
        await _emit_audit("mpesa_payment_confirmed" if rc == 0 else "mpesa_payment_failed", f"m_{meta.get('PhoneNumber', '')}", {"result_code": rc})
    except: pass
    return JSONResponse(content={"ResultCode": 0, "ResultDesc": "Accepted"})

@app.get("/api/v1/payments/summary")
async def get_payments_summary():
    confirmed = [p for p in _PAYMENT_LEDGER.values() if p.get("result_code") == 0]
    return {"total_confirmed_payments": len(confirmed), "total_revenue_ksh": sum(p.get("amount", 0) for p in confirmed), "recent_payments": list(_PAYMENT_LEDGER.values())[-10:]}

@app.get("/api/v1/decisions")
async def get_decisions(limit: int = Query(20, le=100), approved_only: bool = False, channel: Optional[str] = None):
    filtered = _DECISION_LOG[:]
    if approved_only: filtered = [d for d in filtered if d["approved"]]
    if channel: filtered = [d for d in filtered if d["channel"] == channel]
    total = len(_DECISION_LOG); app_count = sum(1 for d in _DECISION_LOG if d["approved"])
    return {"total_decisions": total, "total_approved": app_count, "total_declined": total - app_count, "approval_rate_pct": round((app_count / total * 100), 1) if total > 0 else 0.0, "decisions": list(reversed(filtered))[:limit]}

@app.get("/api/v1/decisions/summary")
async def get_decisions_summary():
    total = len(_DECISION_LOG); approved = [d for d in _DECISION_LOG if d["approved"]]
    by_ch = {"whatsapp": 0, "ussd": 0, "sms": 0, "api": 0}
    by_tier = {"micro": 0, "small": 0, "medium": 0, "declined": 0}
    for d in _DECISION_LOG:
        if d["channel"] in by_ch: by_ch[d["channel"]] += 1
        t = d["agent_2_output"]["credit_tier"] if d["approved"] else "declined"
        if t in by_tier: by_tier[t] += 1
    return {"system_summary": {"total_decisions_all_time": total, "total_approved": len(approved), "total_declined": total - len(approved), "approval_rate_pct": round((len(approved)/total*100),1) if total>0 else 0.0, "bias_proxies_removed_total": sum(len(d["agent_1_output"]["bias_proxies_removed"]) for d in _DECISION_LOG)}, "by_channel": by_ch, "by_tier": by_tier}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check(): return {"healthy": True, "checks": {"api": True, "gemini_configured": bool(GEMINI_API_KEY)}}

@app.get("/")
def read_root():
    return HTMLResponse(f"""<!DOCTYPE html><html><head><title>Kipaji AI Dashboard</title><style>body{{font-family:sans-serif;margin:40px;background:#f4f4f9}}.c{{background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);max-width:600px}}</style></head><body><div class="c"><h1>Kipaji Core AI Dashboard</h1><p>Status: <b>ONLINE</b></p><p>Version: 4.0.0</p><p>Total AI Decisions: <b>{len(_DECISION_LOG)}</b></p><hr><p><a href="/docs">📚 API Docs</a></p><p><a href="/api/v1/decisions">📊 Decision Log</a></p><p><a href="/api/v1/decisions/summary">📈 Summary</a></p></div></body></html>""")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")