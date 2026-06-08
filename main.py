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
- google-genai >= 1.0.0 compatibility
"""
import os
import json
import logging
import asyncio
import base64
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
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
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("KipajiCoreAI")

# =============================================================================
# CONFIGURATION
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# M-PESA DARAJA API CONFIGURATION
MPESA_CONSUMER_KEY = os.getenv("MPESA_CONSUMER_KEY")
MPESA_CONSUMER_SECRET = os.getenv("MPESA_CONSUMER_SECRET")
MPESA_SHORTCODE = os.getenv("MPESA_SHORTCODE", "174379")  # Sandbox default
MPESA_PASSKEY = os.getenv("MPESA_PASSKEY")
MPESA_CALLBACK_URL = os.getenv("MPESA_CALLBACK_URL", "https://kipaji-ai.onrender.com/api/v1/mpesa/callback")
MPESA_ENV = os.getenv("MPESA_ENV", "sandbox")

MPESA_BASE_URL = (
    "https://api.safaricom.co.ke" if MPESA_ENV == "production" 
    else "https://sandbox.safaricom.co.ke"
)

# =============================================================================
# STATE & LEDGERS
# =============================================================================
_PAYMENT_LEDGER: Dict[str, Dict[str, Any]] = {}

_DECISION_LOG: List[Dict[str, Any]] = []
_DECISION_LOG_MAX = 500

def log_credit_decision(
    merchant_id: str,
    channel: str,
    language: str,
    sanitized: SanitizedInput,
    decision: CreditDecision,
    velocity_metrics: Dict[str, float]
):
    """Builds and appends a structured audit log entry for every AI decision."""
    timestamp = datetime.utcnow().isoformat()
    hash_input = f"{merchant_id}{timestamp}"
    decision_id = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    
    entry = {
        "decision_id": decision_id,
        "timestamp": timestamp,
        "merchant_id": merchant_id,
        "channel": channel,
        "language_detected": language,
        "agent_1_output": {
            "overall_confidence": sanitized.overall_confidence,
            "trade_events_extracted": len(sanitized.detected_trade_events),
            "bias_proxies_removed": sanitized.bias_proxy_removed,
            "low_confidence_events": sanitized.low_confidence_count
        },
        "confidence_gate_triggered": decision.confidence_gate_triggered,
        "agent_2_output": {
            "credit_tier": decision.credit_tier,
            "approved_local_ksh": decision.approved_local,
            "velocity_score": decision.velocity_score,
            "consistency_score": decision.consistency_score,
            "interest_rate_monthly_pct": decision.interest_rate_monthly,
            "repayment_days": decision.repayment_days
        },
        "velocity_metrics_at_decision": velocity_metrics,
        "decision_reason": decision.decision_reason,
        "audit_trail": decision.audit_trail,
        "approved": decision.approved
    }
    
    _DECISION_LOG.append(entry)
    if len(_DECISION_LOG) > _DECISION_LOG_MAX:
        _DECISION_LOG.pop(0)

# =============================================================================
# CLIENTS
# =============================================================================
ai_client = None
try:
    if GEMINI_API_KEY:
        from google import genai
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini AI client initialised")
    else:
        logger.warning("GEMINI_API_KEY missing - rule-based fallback will be used")
except Exception as e:
    logger.error(f"Gemini client init failed: {e}")

db = None
try:
    from google.cloud import firestore
    db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
    logger.info("Firestore initialised")
except Exception as e:
    logger.warning(f"Firestore unavailable - in-memory ledger active: {e}")

redis_client = None
try:
    import redis as redis_lib
    redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
    redis_client.ping()
    logger.info("Redis initialised")
except Exception as e:
    logger.warning(f"Redis unavailable - in-memory USSD sessions active: {e}")

# =============================================================================
# APP
# =============================================================================
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
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

# =============================================================================
# TELEMETRY HUB (WITH HISTORY CACHE)
# =============================================================================
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
        telemetry_history.append(message)
        if len(telemetry_history) > MAX_HISTORY:
            telemetry_history.pop(0)
            
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

# =============================================================================
# DATA MODELS
# =============================================================================
class GatewayMessage(BaseModel):
    merchant_id: str = Field(..., min_length=3, max_length=64)
    phone_number: str = Field(..., pattern=r"^\+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1, max_length=2000)

# =============================================================================
# HELPERS
# =============================================================================
_SWAHILI_MARKERS = {"nimeuza", "niliuza", "nilinunua", "biashara", "bei", "leo", "jana", "shilingi", "pesa", "faida", "hasara", "mteja", "wanunuzi", "soko", "nimelipa", "ninapata", "shs", "ksh"}
_SHENG_MARKERS = {"niko na", "nimefanya", "nilifanya", "ilikuwa", "imekuwa", "chapaa", "mkwanja", "mdo", "ka", "fiti"}

def detect_language(text: str) -> str:
    tokens = set(text.lower().split())
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
    if not revenue_events:
        return 0.0
    best = max(revenue_events, key=lambda e: e.confidence)
    return best.amount_ksh

def _format_mpesa_phone(phone: str) -> str:
    """Format phone number to 254XXXXXXXXX for M-Pesa API."""
    clean = phone.replace("+", "").replace(" ", "").replace("-", "")
    if clean.startswith("254"):
        return clean
    if clean.startswith("0"):
        return "254" + clean[1:]
    return "254" + clean

def _generate_mpesa_password(shortcode: str, passkey: str, timestamp: str) -> str:
    """Generate base64-encoded password for M-Pesa STK Push."""
    password_str = f"{shortcode}{passkey}{timestamp}"
    return base64.b64encode(password_str.encode()).decode()

# =============================================================================
# M-PESA DARAJA API FUNCTIONS
# =============================================================================
async def get_mpesa_access_token() -> str:
    """Fetches OAuth access token from Safaricom Daraja API."""
    if not MPESA_CONSUMER_KEY or not MPESA_CONSUMER_SECRET:
        raise ValueError("M-Pesa credentials not configured")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{MPESA_BASE_URL}/oauth/v1/generate?grant_type=client_credentials",
            auth=(MPESA_CONSUMER_KEY, MPESA_CONSUMER_SECRET),
            headers={"Accept": "application/json"}
        )
        if response.status_code != 200:
            raise Exception(f"M-Pesa token fetch failed: {response.status_code} - {response.text}")
        data = response.json()
        if "access_token" not in data:
            raise Exception(f"M-Pesa token response missing access_token: {data}")
        return data["access_token"]

async def trigger_mpesa_stk_push(phone_number: str, merchant_id: str, amount: int = 50) -> dict:
    """Triggers M-Pesa STK Push (Paybill) for the processing fee."""
    try:
        access_token = await get_mpesa_access_token()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        password = _generate_mpesa_password(MPESA_SHORTCODE, MPESA_PASSKEY, timestamp)
        formatted_phone = _format_mpesa_phone(phone_number)
        
        payload = {
            "BusinessShortCode": MPESA_SHORTCODE,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": amount,
            "PartyA": formatted_phone,
            "PartyB": MPESA_SHORTCODE,
            "PhoneNumber": formatted_phone,
            "CallBackURL": MPESA_CALLBACK_URL,
            "AccountReference": f"KIPAJI-{merchant_id[:8].upper()}",
            "TransactionDesc": "Kipaji Credit Processing Fee"
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{MPESA_BASE_URL}/mpesa/stkpush/v1/processrequest",
                json=payload,
                headers=headers
            )
            result = response.json()
            logger.info(f"[{merchant_id}] M-Pesa STK Push triggered: {result.get('ResponseCode', 'N/A')}")
            return result
            
    except Exception as e:
        logger.error(f"[{merchant_id}] M-Pesa STK Push failed: {e}")
        return {"error": str(e), "triggered": False}

# =============================================================================
# WEBSOCKET
# =============================================================================
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_hub.disconnect(websocket)

# =============================================================================
# MAIN GATEWAY (WITH M-PESA FEE COLLECTION & DECISION LOGGING)
# =============================================================================
@app.post("/api/v1/gateway")
async def inbound_telecom_gateway(payload: GatewayMessage, background_tasks: BackgroundTasks):
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

    background_tasks.add_task(
        log_credit_decision,
        merchant_id=merchant_id,
        channel=payload.channel,
        language=user_lang,
        sanitized=sanitized,
        decision=decision,
        velocity_metrics=velocity_metrics
    )

    # M-PESA FEE COLLECTION: Fire-and-forget on approved decisions
    fee_collection_status = "not_applicable"
    if decision.approved and MPESA_CONSUMER_KEY and MPESA_PASSKEY:
        background_tasks.add_task(
            trigger_mpesa_stk_push,
            phone_number=payload.phone_number,
            merchant_id=merchant_id,
            amount=50
        )
        fee_collection_status = "initiated"

    response = {
        "status": "SUCCESS",
        "merchant_id": merchant_id,
        "language_detected": user_lang,
        "credit_decision": decision.model_dump(),
        "velocity_metrics": velocity_metrics,
        "response_message": decision.response_message,
        "fee_collection": fee_collection_status,
    }
    
    return response

# =============================================================================
# USSD GATEWAY (FIXED: PlainTextResponse)
# =============================================================================
@app.post("/api/v1/ussd", response_model=None)
async def ussd_gateway(request: Request, background_tasks: BackgroundTasks):
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
        return PlainTextResponse(
            "CON Karibu Kipaji!\n"
            "Tuambie biashara yako ya leo.\n"
            "Mfano: Niliwauzia wateja mchele 3kg kwa 150 kila moja\n\n"
            "Andika ujumbe wako: "
        )

    if step == 1:
        trade_message = text.strip()
        if len(trade_message) < 5:
            return PlainTextResponse("END Ujumbe mfupi sana. Tafadhali jaribu tena. *384#")

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

        # Log decision BEFORE returning response
        background_tasks.add_task(
            log_credit_decision,
            merchant_id=merchant_id,
            channel="ussd",
            language=user_lang,
            sanitized=sanitized,
            decision=decision,
            velocity_metrics=velocity_metrics
        )

        if decision.confidence_gate_triggered:
            session["step"] = 2
            session["partial_message"] = trade_message
            await save_ussd_session(session_id, session, redis_client=redis_client)
            clarification = decision.response_message[:160]
            return PlainTextResponse(f"CON {clarification}\n\nJibu: ")

        extracted_amount = _extract_primary_trade_amount(sanitized)
        bg_tasks = [
            save_trade_interaction(
                merchant_id=merchant_id,
                extracted_trade_amount=extracted_amount,
                event_type=(sanitized.detected_trade_events[0].event_type
                            if sanitized.detected_trade_events else "unknown"),
                decision_dict=decision.model_dump(),
                bias_removed=sanitized.bias_proxy_removed,
                language=user_lang,
                db=db,
            ),
            _emit_audit("ussd_credit_decision", merchant_id,
                       {"tier": decision.credit_tier, "approved_local": decision.approved_local}),
        ]
        
        if decision.approved and MPESA_CONSUMER_KEY and MPESA_PASSKEY:
            bg_tasks.append(trigger_mpesa_stk_push(phone_number, merchant_id, 50))
            
        for task in bg_tasks:
            background_tasks.add_task(task)

        await clear_ussd_session(session_id, redis_client=redis_client)
        result_msg = decision.response_message[:160]
        return PlainTextResponse(f"END {result_msg}")

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

        background_tasks.add_task(
            log_credit_decision,
            merchant_id=merchant_id,
            channel="ussd",
            language=user_lang,
            sanitized=sanitized,
            decision=decision,
            velocity_metrics=velocity_metrics
        )

        extracted_amount = _extract_primary_trade_amount(sanitized)
        background_tasks.add_task(
            save_trade_interaction, merchant_id, extracted_amount,
            (sanitized.detected_trade_events[0].event_type if sanitized.detected_trade_events else "unknown"),
            decision.model_dump(), sanitized.bias_proxy_removed, user_lang, db,
        )

        await clear_ussd_session(session_id, redis_client=redis_client)
        return PlainTextResponse(f"END {decision.response_message[:160]}")

    await clear_ussd_session(session_id, redis_client=redis_client)
    return PlainTextResponse("END Kuna tatizo. Tafadhali piga *384# tena.")

# =============================================================================
# M-PESA CALLBACK ENDPOINT
# =============================================================================
@app.post("/api/v1/mpesa/callback")
async def mpesa_callback(request: Request):
    """Handles Daraja API callback for STK Push payment confirmation."""
    try:
        body = await request.json()
        stk_callback = body.get("Body", {}).get("stkCallback", {})
        
        merchant_request_id = stk_callback.get("MerchantRequestID")
        checkout_request_id = stk_callback.get("CheckoutRequestID")
        result_code = stk_callback.get("ResultCode")
        result_desc = stk_callback.get("ResultDesc")
        
        callback_metadata = stk_callback.get("CallbackMetadata", {}).get("Item", [])
        metadata_dict = {item["Name"]: item["Value"] for item in callback_metadata} if callback_metadata else {}
        
        amount = metadata_dict.get("Amount")
        mpesa_receipt = metadata_dict.get("MpesaReceiptNumber")
        phone = metadata_dict.get("PhoneNumber")
        
        merchant_id = f"m_{phone}" if phone else "unknown"
        
        payment_record = {
            "merchant_request_id": merchant_request_id,
            "checkout_request_id": checkout_request_id,
            "result_code": result_code,
            "result_desc": result_desc,
            "amount": amount,
            "mpesa_receipt": mpesa_receipt,
            "phone": phone,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if result_code == 0:
            if mpesa_receipt:
                _PAYMENT_LEDGER[mpesa_receipt] = payment_record
            await _emit_audit("mpesa_payment_confirmed", merchant_id, payment_record)
            logger.info(f"[{merchant_id}] M-Pesa payment confirmed: {mpesa_receipt} - KSH {amount}")
        else:
            await _emit_audit("mpesa_payment_failed", merchant_id, payment_record)
            logger.warning(f"[{merchant_id}] M-Pesa payment failed: {result_desc}")
            
    except Exception as e:
        logger.error(f"M-Pesa callback processing error: {e}")
    
    # Safaricom requires this exact response format
    return JSONResponse(content={"ResultCode": 0, "ResultDesc": "Accepted"})

# =============================================================================
# PAYMENTS & DECISIONS ENDPOINTS (EVIDENCE FOR JUDGES)
# =============================================================================
@app.get("/api/v1/payments/summary")
async def get_payments_summary():
    """Returns revenue evidence for hackathon judges."""
    confirmed = [p for p in _PAYMENT_LEDGER.values() if p.get("result_code") == 0]
    total_revenue = sum(p.get("amount", 0) for p in confirmed)
    
    return {
        "total_confirmed_payments": len(confirmed),
        "total_revenue_ksh": round(total_revenue, 2),
        "recent_payments": list(_PAYMENT_LEDGER.values())[-10:],
        "currency": "KSH",
        "fee_per_transaction": 50,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/api/v1/decisions")
async def get_audit_log(
    limit: int = Query(20, le=100),
    approved_only: bool = False,
    channel: Optional[str] = None
):
    filtered_log = _DECISION_LOG[:]
    
    if approved_only:
        filtered_log = [d for d in filtered_log if d["approved"]]
        
    if channel:
        filtered_log = [d for d in filtered_log if d["channel"] == channel]
        
    filtered_log = list(reversed(filtered_log))
    
    total_decisions = len(_DECISION_LOG)
    total_approved = sum(1 for d in _DECISION_LOG if d["approved"])
    total_declined = total_decisions - total_approved
    approval_rate_pct = round((total_approved / total_decisions * 100), 1) if total_decisions > 0 else 0.0
    
    return {
        "total_decisions": total_decisions,
        "total_approved": total_approved,
        "total_declined": total_declined,
        "approval_rate_pct": approval_rate_pct,
        "decisions": filtered_log[:limit]
    }

@app.get("/api/v1/decisions/summary")
async def get_decisions_summary():
    total = len(_DECISION_LOG)
    approved = sum(1 for d in _DECISION_LOG if d["approved"])
    declined = total - approved
    approval_rate = round((approved / total * 100), 1) if total > 0 else 0.0
    
    conf_gates = sum(1 for d in _DECISION_LOG if d["confidence_gate_triggered"])
    
    approved_decisions = [d for d in _DECISION_LOG if d["approved"]]
    avg_velocity = round(sum(d["agent_2_output"]["velocity_score"] for d in approved_decisions) / len(approved_decisions), 2) if approved_decisions else 0.0
    avg_amount = round(sum(d["agent_2_output"]["approved_local_ksh"] for d in approved_decisions) / len(approved_decisions), 2) if approved_decisions else 0.0
    
    bias_total = sum(len(d["agent_1_output"]["bias_proxies_removed"]) for d in _DECISION_LOG)
    
    by_channel = {"whatsapp": 0, "ussd": 0, "sms": 0, "api": 0}
    by_tier = {"micro": 0, "small": 0, "medium": 0, "declined": 0}
    by_language = {"sw": 0, "en": 0, "mixed": 0}
    
    for d in _DECISION_LOG:
        ch = d["channel"]
        if ch in by_channel: by_channel[ch] += 1
        
        tier = d["agent_2_output"]["credit_tier"]
        if tier in by_tier: by_tier[tier] += 1
        
        lang = d.get("language_detected", "en")
        if lang in by_language:
            by_language[lang] += 1
        else:
            by_language["mixed"] += 1

    unique_bias_removals = []
    seen = set()
    for d in reversed(_DECISION_LOG):
        proxies = d["agent_1_output"]["bias_proxies_removed"]
        if proxies:
            tup = tuple(proxies)
            if tup not in seen:
                seen.add(tup)
                unique_bias_removals.append(proxies)
                if len(unique_bias_removals) == 5:
                    break
                    
    return {
        "system_summary": {
            "total_decisions_all_time": total,
            "total_approved": approved,
            "total_declined": declined,
            "approval_rate_pct": approval_rate,
            "confidence_gates_triggered": conf_gates,
            "avg_velocity_score": avg_velocity,
            "avg_approved_amount_ksh": avg_amount,
            "bias_proxies_removed_total": bias_total
        },
        "by_channel": by_channel,
        "by_tier": by_tier,
        "by_language": by_language,
        "recent_bias_removals": unique_bias_removals
    }

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================
@app.get("/")
def read_root():
    return {
        "service": "Kipaji Core AI",
        "version": "4.0.0",
        "status": "ONLINE",
        "gemini": "connected" if ai_client else "unavailable (rule-based fallback active)",
        "firestore": "connected" if db else "unavailable (in-memory fallback active)",
        "redis": "connected" if redis_client else "unavailable (in-memory sessions active)",
        "mpesa": "configured" if (MPESA_CONSUMER_KEY and MPESA_PASSKEY) else "not_configured",
        "total_ai_decisions_logged": len(_DECISION_LOG),
    }

# FIXED: Accept both GET and HEAD requests for UptimeRobot
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    checks = {
        "api": True,
        "gemini_configured": bool(GEMINI_API_KEY),
        "mpesa_configured": bool(MPESA_CONSUMER_KEY and MPESA_PASSKEY),
    }
    return {"healthy": True, "checks": checks, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v1/merchant/{merchant_id}/history")
async def get_merchant_history(merchant_id: str):
    data = await get_merchant_data(merchant_id, db=db)
    metrics = calculate_velocity_metrics(data["history"])
    return {
        "merchant_id": merchant_id,
        "trade_history_count": len(data["history"]),
        "recent_history": data["history"][-10:],
        "velocity_metrics": metrics,
        "profile": data["profile"],
    }

# =============================================================================
# LIVE DASHBOARD
# =============================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")