"""
main.py — Kipaji Core AI v4.0 (Production-Ready Edition)
"""
import os
import json
import base64
import logging
import asyncio
import hashlib
import httpx
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
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
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("KipajiCoreAI")

# CONFIGURATION
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# M-PESA CONFIGURATION
MPESA_CONSUMER_KEY = os.getenv("MPESA_CONSUMER_KEY")
MPESA_CONSUMER_SECRET = os.getenv("MPESA_CONSUMER_SECRET")
MPESA_SHORTCODE = os.getenv("MPESA_SHORTCODE")
MPESA_PASSKEY = os.getenv("MPESA_PASSKEY")
MPESA_CALLBACK_URL = os.getenv("MPESA_CALLBACK_URL")
MPESA_ENV = os.getenv("MPESA_ENV", "sandbox")

MPESA_BASE_URL = "https://sandbox.safaricom.co.ke" if MPESA_ENV == "sandbox" else "https://api.safaricom.co.ke"

# In-memory ledger for hackathon demo (in production, this would be Firestore/SQL)
_PAYMENT_LEDGER: Dict[str, Dict] = {}

# PERSISTENT AUDIT LOG FOR HACKATHON JUDGES
_DECISION_LOG: List[Dict[str, Any]] = []
_DECISION_LOG_MAX = 500

def log_credit_decision(merchant_id, channel, language, sanitized, decision, velocity_metrics):
    """
    Logs a credit decision to the persistent in-memory audit log.
    This survives WebSocket disconnections and serves as proof of continuous AI operation.
    """
    timestamp = datetime.utcnow().isoformat()
    decision_id = hashlib.sha256(f"{merchant_id}{timestamp}".encode()).hexdigest()[:8]
    
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

# CLIENTS
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

# APP
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
    allow_methods=["GET", "POST", "HEAD"],
    allow_headers=["*"],
)

# TELEMETRY HUB (WITH HISTORY CACHE)
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

# --- M-PESA INTEGRATION FUNCTIONS ---

async def get_mpesa_access_token() -> str:
    """Fetches an OAuth access token from the Safaricom Daraja API."""
    url = f"{MPESA_BASE_URL}/oauth/v1/generate?grant_type=client_credentials"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                auth=(MPESA_CONSUMER_KEY, MPESA_CONSUMER_SECRET),
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data["access_token"]
        except Exception as e:
            logger.error(f"Failed to fetch M-Pesa access token: {e}")
            raise Exception(f"M-Pesa token fetch failed: {str(e)}")

async def trigger_mpesa_stk_push(phone_number: str, merchant_id: str, amount: int = 50) -> dict:
    """Triggers an M-Pesa STK Push to collect the KSH 50 processing fee."""
    try:
        token = await get_mpesa_access_token()
        
        # Format phone number to 254XXXXXXXXX
        phone = phone_number.strip().replace(" ", "")
        if phone.startswith("+"):
            phone = phone[1:]
        if phone.startswith("254"):
            phone = phone[3:]
        elif phone.startswith("0"):
            phone = phone[1:]
        phone = "254" + phone
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        password_str = f"{MPESA_SHORTCODE}{MPESA_PASSKEY}{timestamp}"
        password = base64.b64encode(password_str.encode()).decode('utf-8')
        
        url = f"{MPESA_BASE_URL}/mpesa/stkpush/v1/processrequest"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "BusinessShortCode": MPESA_SHORTCODE,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": amount,
            "PartyA": phone,
            "PartyB": MPESA_SHORTCODE,
            "PhoneNumber": phone,
            "CallBackURL": MPESA_CALLBACK_URL,
            "AccountReference": f"KIPAJI-{merchant_id[:8].upper()}",
            "TransactionDesc": "Kipaji Credit Processing Fee"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=15.0)
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Error triggering M-Pesa STK Push for {merchant_id}: {e}")
        return {"error": str(e), "triggered": False}

# DATA MODELS
class GatewayMessage(BaseModel):
    merchant_id: str = Field(..., min_length=3, max_length=64)
    # FIXED: Escaped the '+' in the regex pattern
    phone_number: str = Field(..., pattern=r"^\+?[0-9]{9,15}$")
    channel: str = Field(..., pattern=r"^(whatsapp|sms|ussd|api)$")
    message_body: str = Field(..., min_length=1, max_length=2000)

# HELPERS
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

# WEBSOCKET
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_hub.disconnect(websocket)

# MAIN GATEWAY
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

    # Log to persistent audit log for hackathon judges
    background_tasks.add_task(
        log_credit_decision,
        merchant_id=merchant_id,
        channel=payload.channel,
        language=user_lang,
        sanitized=sanitized,
        decision=decision,
        velocity_metrics=velocity_metrics
    )

    # M-Pesa Fee Collection Logic
    fee_collection_status = "not_applicable"
    if decision.approved:
        fee_collection_status = "initiated"
        background_tasks.add_task(
            trigger_mpesa_stk_push,
            phone_number=payload.phone_number,
            merchant_id=payload.merchant_id,
            amount=50
        )

    return {
        "status": "SUCCESS",
        "merchant_id": merchant_id,
        "language_detected": user_lang,
        "credit_decision": decision.model_dump(),
        "velocity_metrics": velocity_metrics,
        "response_message": decision.response_message,
        "fee_collection": fee_collection_status,
    }

# M-PESA CALLBACK ENDPOINT
@app.post("/api/v1/mpesa/callback")
async def mpesa_callback(request: Request):
    try:
        body = await request.json()
        # Daraja wraps the callback in a "Body" -> "stkCallback" structure
        callback_data = body.get("Body", {}).get("stkCallback", body)
        
        merchant_request_id = callback_data.get("MerchantRequestID")
        checkout_request_id = callback_data.get("CheckoutRequestID")
        result_code = callback_data.get("ResultCode")
        result_desc = callback_data.get("ResultDesc")
        
        # Extract merchant_id from AccountReference
        account_ref = callback_data.get("AccountReference", "")
        extracted_merchant_id = account_ref.replace("KIPAJI-", "") if account_ref.startswith("KIPAJI-") else account_ref
        
        amount = 0
        mpesa_receipt_number = ""
        phone_number = ""
        
        if "CallbackMetadata" in callback_data:
            for item in callback_data["CallbackMetadata"].get("Item", []):
                if item["Name"] == "Amount":
                    amount = item.get("Value", 0)
                elif item["Name"] == "MpesaReceiptNumber":
                    mpesa_receipt_number = item.get("Value", "")
                elif item["Name"] == "PhoneNumber":
                    phone_number = item.get("Value", "")
                    
        payment_record = {
            "MerchantRequestID": merchant_request_id,
            "CheckoutRequestID": checkout_request_id,
            "ResultCode": result_code,
            "ResultDesc": result_desc,
            "Amount": amount,
            "MpesaReceiptNumber": mpesa_receipt_number,
            "PhoneNumber": phone_number,
            "AccountReference": account_ref,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result_code == 0:
            if mpesa_receipt_number:
                _PAYMENT_LEDGER[mpesa_receipt_number] = payment_record
            await _emit_audit("mpesa_payment_confirmed", extracted_merchant_id, payment_record)
        else:
            await _emit_audit("mpesa_payment_failed", extracted_merchant_id, payment_record)
            
    except Exception as e:
        logger.error(f"Error processing M-Pesa callback: {e}")
        
    # Safaricom requires this exact response to acknowledge the callback
    return JSONResponse(content={"ResultCode": 0, "ResultDesc": "Accepted"})

# PAYMENTS SUMMARY ENDPOINT (For Hackathon Judges)
@app.get("/api/v1/payments/summary")
async def get_payments_summary():
    payments = list(_PAYMENT_LEDGER.values())
    total_revenue = sum(p.get("Amount", 0) for p in payments)
    recent_payments = payments[-10:]
    
    return {
        "total_confirmed_payments": len(payments),
        "total_revenue_ksh": total_revenue,
        "recent_payments": recent_payments
    }

# USSD GATEWAY
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
        
        # Log to persistent audit log for hackathon judges
        background_tasks.add_task(
            log_credit_decision,
            merchant_id=merchant_id,
            channel="ussd",
            language=user_lang,
            sanitized=sanitized,
            decision=decision,
            velocity_metrics=velocity_metrics
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
        
        # Log to persistent audit log for hackathon judges
        background_tasks.add_task(
            log_credit_decision,
            merchant_id=merchant_id,
            channel="ussd",
            language=user_lang,
            sanitized=sanitized,
            decision=decision,
            velocity_metrics=velocity_metrics
        )

        await clear_ussd_session(session_id, redis_client=redis_client)
        return _ussd_end(decision.response_message[:160])

    await clear_ussd_session(session_id, redis_client=redis_client)
    return _ussd_end("Kuna tatizo. Tafadhali piga *384# tena.")

# FIXED: Return PlainTextResponse to prevent JSON quote wrapping
def _ussd_continue(text: str) -> PlainTextResponse:
    return PlainTextResponse(f"CON {text}")

def _ussd_end(text: str) -> PlainTextResponse:
    return PlainTextResponse(f"END {text}")

# UTILITY ENDPOINTS
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Kipaji Core AI Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
                h1 {{ color: #2c3e50; }}
                .stat {{ font-size: 1.5em; color: #2980b9; font-weight: bold; }}
                .container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 600px; }}
                a {{ color: #2980b9; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Kipaji Core AI Dashboard</h1>
                <p>Service Status: <span class="stat">ONLINE</span></p>
                <p>Version: <span class="stat">4.0.0</span></p>
                <p>Total AI Decisions: <span class="stat">{len(_DECISION_LOG)}</span></p>
                <hr>
                <p><a href="/docs">📚 API Documentation (Swagger)</a></p>
                <p><a href="/api/v1/decisions">📊 View Decision Log</a></p>
                <p><a href="/api/v1/decisions/summary">📈 View Decision Summary</a></p>
            </div>
        </body>
    </html>
    """
    return html_content

# FIXED: Accept both GET and HEAD requests for UptimeRobot
@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    checks = {
        "api": True,
        "gemini_configured": bool(GEMINI_API_KEY),
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

# JUDGE AUDIT ENDPOINTS
@app.get("/api/v1/decisions")
async def get_decisions(
    limit: int = Query(20, ge=1, le=100), 
    approved_only: bool = False, 
    channel: Optional[str] = None
):
    total_decisions = len(_DECISION_LOG)
    total_approved = sum(1 for d in _DECISION_LOG if d.get("approved"))
    total_declined = total_decisions - total_approved
    approval_rate_pct = round((total_approved / total_decisions) * 100, 1) if total_decisions > 0 else 0.0
    
    filtered = _DECISION_LOG[:]
    if approved_only:
        filtered = [d for d in filtered if d.get("approved")]
    if channel:
        filtered = [d for d in filtered if d.get("channel") == channel]
        
    # Most recent first
    filtered.reverse()
    
    return {
        "total_decisions": total_decisions,
        "total_approved": total_approved,
        "total_declined": total_declined,
        "approval_rate_pct": approval_rate_pct,
        "decisions": filtered[:limit]
    }

@app.get("/api/v1/decisions/summary")
async def get_decisions_summary():
    total = len(_DECISION_LOG)
    approved = [d for d in _DECISION_LOG if d.get("approved")]
    declined = [d for d in _DECISION_LOG if not d.get("approved")]
    
    total_approved = len(approved)
    total_declined = len(declined)
    approval_rate = round((total_approved / total) * 100, 1) if total > 0 else 0.0
    
    confidence_gates = sum(1 for d in _DECISION_LOG if d.get("confidence_gate_triggered"))
    
    velocity_scores = [d.get("agent_2_output", {}).get("velocity_score", 0) for d in _DECISION_LOG]
    avg_velocity = round(sum(velocity_scores) / len(velocity_scores), 2) if velocity_scores else 0.0
    
    approved_amounts = [
        d.get("agent_2_output", {}).get("approved_local_ksh", 0) 
        for d in approved 
        if d.get("agent_2_output", {}).get("approved_local_ksh") is not None
    ]
    avg_approved_amount = round(sum(approved_amounts) / len(approved_amounts), 2) if approved_amounts else 0.0
    
    bias_total = sum(len(d.get("agent_1_output", {}).get("bias_proxies_removed", [])) for d in _DECISION_LOG)
    
    by_channel = {"whatsapp": 0, "ussd": 0, "sms": 0, "api": 0}
    for d in _DECISION_LOG:
        ch = d.get("channel")
        if ch in by_channel:
            by_channel[ch] += 1
            
    by_tier = {"micro": 0, "small": 0, "medium": 0, "declined": 0}
    for d in _DECISION_LOG:
        if not d.get("approved"):
            by_tier["declined"] += 1
        else:
            tier = d.get("agent_2_output", {}).get("credit_tier")
            if tier in by_tier:
                by_tier[tier] += 1
                
    by_language = {"sw": 0, "en": 0, "mixed": 0}
    for d in _DECISION_LOG:
        lang = d.get("language_detected")
        if lang in by_language:
            by_language[lang] += 1
            
    # recent_bias_removals: last 5 unique bias_proxy_removed lists across all decisions
    seen = []
    seen_tuples = []
    for d in reversed(_DECISION_LOG):
        bpr = d.get("agent_1_output", {}).get("bias_proxies_removed", [])
        t_bpr = tuple(bpr)
        if t_bpr not in seen_tuples:
            seen_tuples.append(t_bpr)
            seen.append(bpr)
        if len(seen) == 5:
            break
            
    return {
        "system_summary": {
            "total_decisions_all_time": total,
            "total_approved": total_approved,
            "total_declined": total_declined,
            "approval_rate_pct": approval_rate,
            "confidence_gates_triggered": confidence_gates,
            "avg_velocity_score": avg_velocity,
            "avg_approved_amount_ksh": avg_approved_amount,
            "bias_proxies_removed_total": bias_total
        },
        "by_channel": by_channel,
        "by_tier": by_tier,
        "by_language": by_language,
        "recent_bias_removals": seen
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")