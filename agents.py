"""
agents.py — Kipaji-AI core agent pipeline (Groq Llama 3.3 Edition)
"""
import json
import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger("KipajiAgents")

MODEL_ID = "llama-3.3-70b-versatile"

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

async def _call_llm(client, prompt: str, system: str = "") -> str:
    """Wraps synchronous Groq call in asyncio.to_thread."""
    def _blocking_call():
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    return await asyncio.to_thread(_blocking_call)

BIAS_GUARDRAIL_SYSTEM = """
You are a bias-mitigation filter for an AI credit system serving African informal economy workers.
Extract typed trade events and remove demographic proxy variables (gender, ethnicity, location).
Return ONLY structured JSON.
TRADE EVENT TYPES: revenue, expense, inventory_in, debt_repayment, receivable, unknown.
JSON Structure:
{
  "cleaned_text": "sanitized input",
  "detected_trade_events": [{"event_type": "...", "amount_ksh": null or number, "frequency_signal": null or "daily|weekly|market_day|irregular", "confidence": 0.0-1.0, "raw_utterance": "...", "bias_flags": []}],
  "language": "sw|en|mixed",
  "low_confidence_count": number,
  "overall_confidence": 0.0-1.0,
  "bias_proxy_removed": ["list of proxies removed"]
}
"""

async def run_bias_mitigation_guardrail(raw_message: str, merchant_id: str, language: str, ai_client=None) -> SanitizedInput:
    if ai_client is None:
        return _guardrail_fallback(raw_message, language)

    prompt = f"Merchant message (language hint: {language}):\n\"\"\"\n{raw_message}\n\"\"\"\nExtract trade events and apply bias filtering. Return only JSON."
    
    try:
        raw_response = await _call_llm(ai_client, prompt, system=BIAS_GUARDRAIL_SYSTEM)
        clean = raw_response.strip()
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end+1]
        else:
            raise ValueError("No JSON object found")
            
        data = json.loads(clean)
        sanitized = SanitizedInput(**data)
        logger.info(f"[{merchant_id}] Guardrail: {len(sanitized.detected_trade_events)} events, confidence={sanitized.overall_confidence:.2f}")
        return sanitized
    except Exception as e:
        logger.error(f"[{merchant_id}] Guardrail parse error: {e}")
        return _guardrail_fallback(raw_message, language)

def _guardrail_fallback(raw_message: str, language: str) -> SanitizedInput:
    return SanitizedInput(
        cleaned_text=raw_message,
        detected_trade_events=[TradeEvent(event_type="unknown", amount_ksh=None, confidence=0.1, raw_utterance=raw_message[:200], bias_flags=[])],
        language=language, low_confidence_count=1, overall_confidence=0.1, bias_proxy_removed=[]
    )

UNDERWRITER_SYSTEM = """
You are an AI credit underwriter for Kipaji. Base decisions ONLY on trade velocity.
COLD START BOOTSTRAPPING: If avg_daily=0 but current trade event is valid (confidence > 0.6, amount > 0), you MUST approve a "micro" tier loan (e.g., 500 KSH) to bootstrap their credit file.
CREDIT TIERS (KSH): micro: 500-2000, small: 2001-8000, medium: 8001-25000, declined: 0.
INTEREST: micro: 4%, small: 3.5%, medium: 3%. REPAYMENT: 7, 14, 21 days respectively.
Return ONLY JSON:
{
  "approved": true/false, "approved_local": number, "credit_tier": "micro|small|medium|declined",
  "interest_rate_monthly": number, "repayment_days": number, "velocity_score": number 0-100,
  "consistency_score": number 0-1, "confidence_gate_triggered": true/false,
  "decision_reason": "one sentence", "audit_trail": ["list"],
  "response_message": "merchant-facing message in their language (sw/en), warm and respectful"
}
"""

async def run_kipaji_underwriter_core(sanitized: SanitizedInput, history: List[Dict[str, Any]], profile: Dict[str, Any], velocity_metrics: Dict[str, float], merchant_id: str, ai_client=None) -> CreditDecision:
    if sanitized.overall_confidence < 0.65:
        return _low_confidence_decision(sanitized, merchant_id)

    revenue_events = [e for e in sanitized.detected_trade_events if e.event_type in ("revenue", "receivable") and e.amount_ksh]
    prompt = f"""
Merchant ID: {merchant_id} | Language: {sanitized.language}
EVENTS: {json.dumps([e.model_dump() for e in revenue_events])}
HISTORY: {json.dumps(history[-10:])}
METRICS: avg_daily={velocity_metrics.get('avg_daily', 0):.2f}, total_7d={velocity_metrics.get('total_7d', 0):.2f}, consistency={velocity_metrics.get('consistency', 1.0):.2f}
Make a credit decision based solely on velocity signals. Return only JSON.
"""

    if ai_client is None:
        return _rule_based_decision(revenue_events, velocity_metrics, sanitized.language, merchant_id)

    try:
        raw_response = await _call_llm(ai_client, prompt, system=UNDERWRITER_SYSTEM)
        clean = raw_response.strip()
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end+1]
        else:
            raise ValueError("No JSON object found")
            
        data = json.loads(clean)
        data["timestamp"] = datetime.utcnow().isoformat()
        decision = CreditDecision(**data)
        logger.info(f"[{merchant_id}] Decision: {decision.credit_tier} | KSH {decision.approved_local}")
        return decision
    except Exception as e:
        logger.error(f"[{merchant_id}] Underwriter parse error: {e}")
        return _rule_based_decision(revenue_events, velocity_metrics, sanitized.language, merchant_id)

def _low_confidence_decision(sanitized: SanitizedInput, merchant_id: str) -> CreditDecision:
    msg = "Samahani, hatukuelewa vizuri. Tafadhali tuambie: uliuza nini leo na bei ngapi?" if sanitized.language == "sw" else "Sorry, we couldn't understand your message clearly. Could you tell us: what did you sell today and for how much?"
    return CreditDecision(approved=False, approved_local=0, credit_tier="declined", interest_rate_monthly=0, repayment_days=0, velocity_score=0, consistency_score=0, confidence_gate_triggered=True, decision_reason=f"Confidence gate: overall_confidence={sanitized.overall_confidence:.2f}", audit_trail=["Confidence below threshold 0.65"], response_message=msg, timestamp=datetime.utcnow().isoformat())

def _rule_based_decision(revenue_events: list, velocity_metrics: Dict[str, float], language: str, merchant_id: str) -> CreditDecision:
    avg_daily = velocity_metrics.get("avg_daily", 0)
    consistency = velocity_metrics.get("consistency", 0.5)
    total_7d = velocity_metrics.get("total_7d", 0)
    velocity_score = min(100, (avg_daily / 1000) * 40 + (total_7d / 5000) * 40 + consistency * 20)

    if avg_daily >= 3000 and consistency >= 0.6: tier, amount, rate, days = "medium", min(avg_daily * 2.5, 25000), 3.0, 21
    elif avg_daily >= 800 and consistency >= 0.4: tier, amount, rate, days = "small", min(avg_daily * 2.0, 8000), 3.5, 14
    elif avg_daily >= 200 or (total_7d > 0 and len(revenue_events) > 0): tier, amount, rate, days = "micro", min(max(avg_daily * 1.5, 500), 2000), 4.0, 7
    else: tier, amount, rate, days = "declined", 0, 0, 0

    msg = f"Hongera! Umeidhinishiwa mkopo wa KSH {amount:,.0f} kwa siku {days}." if language == "sw" and amount > 0 else f"Approved! KSH {amount:,.0f} credit for {days} days." if amount > 0 else "We need more trading history."
    
    return CreditDecision(approved=amount > 0, approved_local=round(amount, 2), credit_tier=tier, interest_rate_monthly=rate, repayment_days=days, velocity_score=round(velocity_score, 1), consistency_score=round(consistency, 2), confidence_gate_triggered=False, decision_reason=f"Rule-based fallback", audit_trail=[f"avg_daily={avg_daily}"], response_message=msg, timestamp=datetime.utcnow().isoformat())