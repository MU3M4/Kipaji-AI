import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger("KipajiAgents")

MODEL_ID_GEMINI = "gemini-2.0-flash"
MODEL_ID_GROQ = "llama-3.3-70b-versatile"

# --- TYPED CONTRACT ---
class TradeEvent(BaseModel):
    event_type: Literal["revenue", "expense", "inventory_in", "debt_repayment", "receivable", "unknown"]
    amount_ksh: Optional[float] = None
    frequency_signal: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    raw_utterance: str
    bias_flags: List[str] = []

class SanitizedInput(BaseModel):
    cleaned_text: str
    detected_trade_events: List[TradeEvent]
    language: str
    low_confidence_count: int
    overall_confidence: float
    bias_proxy_removed: List[str]

class CreditDecision(BaseModel):
    approved: bool
    approved_local: float
    credit_tier: Literal["micro", "small", "medium", "declined"]
    interest_rate_monthly: float
    repayment_days: int
    velocity_score: float
    consistency_score: float
    confidence_gate_triggered: bool
    decision_reason: str
    audit_trail: List[str]
    response_message: str
    timestamp: str

# --- LLM HELPERS ---
async def _call_gemini(client, prompt: str, system: str = "") -> str:
    from google.genai import types
    def _blocking():
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        config = types.GenerateContentConfig(system_instruction=system, temperature=0.1, max_output_tokens=1500)
        return client.models.generate_content(model=MODEL_ID_GEMINI, contents=contents, config=config).text
    return await asyncio.to_thread(_blocking)

async def _call_groq(client, prompt: str, system: str = "") -> str:
    def _blocking():
        response = client.chat.completions.create(model=MODEL_ID_GROQ, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], temperature=0.1, max_tokens=1500)
        return response.choices[0].message.content
    return await asyncio.to_thread(_blocking)

# --- AGENT 1: BIAS GUARDRAIL (GEMINI) ---
BIAS_GUARDRAIL_SYSTEM = """You are a bias-mitigation filter. Extract trade events and remove demographic proxies. Return ONLY JSON.
{"cleaned_text": "...", "detected_trade_events": [{"event_type": "revenue|expense|inventory_in|debt_repayment|receivable|unknown", "amount_ksh": null, "frequency_signal": null, "confidence": 0.0, "raw_utterance": "...", "bias_flags": []}], "language": "sw|en", "low_confidence_count": 0, "overall_confidence": 0.0, "bias_proxy_removed": []}"""

async def run_bias_mitigation_guardrail(raw_message: str, merchant_id: str, language: str, ai_client=None) -> SanitizedInput:
    if ai_client is None: return _guardrail_fallback(raw_message, language)
    prompt = f"Merchant message:\n{raw_message}\nExtract trade events. Return only JSON."
    try:
        raw = await _call_gemini(ai_client, prompt, system=BIAS_GUARDRAIL_SYSTEM)
        clean = raw.strip()
        start, end = clean.find('{'), clean.rfind('}')
        if start != -1 and end != -1: clean = clean[start:end+1]
        return SanitizedInput(**json.loads(clean))
    except Exception as e:
        logger.error(f"[{merchant_id}] Guardrail error: {e}")
        return _guardrail_fallback(raw_message, language)

def _guardrail_fallback(raw_message: str, language: str) -> SanitizedInput:
    return SanitizedInput(cleaned_text=raw_message, detected_trade_events=[TradeEvent(event_type="unknown", amount_ksh=None, confidence=0.1, raw_utterance=raw_message, bias_flags=[])], language=language, low_confidence_count=1, overall_confidence=0.1, bias_proxy_removed=[])

# --- AGENT 2: UNDERWRITER (GROQ) ---
UNDERWRITER_SYSTEM = """You are an AI credit underwriter. Base decisions ONLY on trade velocity.
COLD START: If history is empty but current trade is valid, approve "micro" (500 KSH).
Return ONLY JSON: {"approved": true, "approved_local": 500, "credit_tier": "micro", "interest_rate_monthly": 4.0, "repayment_days": 7, "velocity_score": 50, "consistency_score": 0.5, "confidence_gate_triggered": false, "decision_reason": "...", "audit_trail": [], "response_message": "Swahili/English message"}"""

async def run_kipaji_underwriter_core(sanitized: SanitizedInput, history: List[Dict], profile: Dict, velocity_metrics: Dict, merchant_id: str, ai_client=None) -> CreditDecision:
    if sanitized.overall_confidence < 0.65: return _low_confidence_decision(sanitized, merchant_id)
    revenue_events = [e for e in sanitized.detected_trade_events if e.event_type in ("revenue", "receivable") and e.amount_ksh]
    prompt = f"EVENTS: {json.dumps([e.model_dump() for e in revenue_events])}\nHISTORY: {json.dumps(history[-10:])}\nMETRICS: {velocity_metrics}\nMake decision. Return JSON."
    
    if ai_client is None: return _rule_based_decision(revenue_events, velocity_metrics, sanitized.language, merchant_id)
    try:
        raw = await _call_groq(ai_client, prompt, system=UNDERWRITER_SYSTEM)
        clean = raw.strip()
        start, end = clean.find('{'), clean.rfind('}')
        if start != -1 and end != -1: clean = clean[start:end+1]
        data = json.loads(clean)
        data["timestamp"] = datetime.utcnow().isoformat()
        return CreditDecision(**data)
    except Exception as e:
        logger.error(f"[{merchant_id}] Underwriter error: {e}")
        return _rule_based_decision(revenue_events, velocity_metrics, sanitized.language, merchant_id)

def _low_confidence_decision(sanitized: SanitizedInput, merchant_id: str) -> CreditDecision:
    msg = "Samahani, hatukuelewa vizuri. Tafadhali tuambie: uliuza nini leo na bei ngapi?" if sanitized.language == "sw" else "Sorry, please tell us what you sold today and for how much."
    return CreditDecision(approved=False, approved_local=0, credit_tier="declined", interest_rate_monthly=0, repayment_days=0, velocity_score=0, consistency_score=0, confidence_gate_triggered=True, decision_reason="Confidence gate", audit_trail=[], response_message=msg, timestamp=datetime.utcnow().isoformat())

def _rule_based_decision(revenue_events: list, velocity_metrics: Dict, language: str, merchant_id: str) -> CreditDecision:
    avg_daily = velocity_metrics.get("avg_daily", 0)
    tier, amount, rate, days = "micro", 500, 4.0, 7
    msg = f"Hongera! Umeidhinishiwa mkopo wa KSH {amount}." if language == "sw" else f"Approved! KSH {amount} credit."
    return CreditDecision(approved=True, approved_local=amount, credit_tier=tier, interest_rate_monthly=rate, repayment_days=days, velocity_score=50, consistency_score=0.5, confidence_gate_triggered=False, decision_reason="Rule-based", audit_trail=[], response_message=msg, timestamp=datetime.utcnow().isoformat())