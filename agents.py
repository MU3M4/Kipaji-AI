"""
agents.py — Kipaji-AI core agent pipeline
"""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger("KipajiAgents")

MODEL_ID = "gemini-1.5-flash"

# ---------------------------------------------------------------------------
# TYPED INTER-AGENT CONTRACT
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# SHARED GEMINI CALL HELPER (FIXED)
# ---------------------------------------------------------------------------

async def _gemini_generate(client, prompt: str, system: str = "") -> str:
    """
    Wraps the synchronous Gemini generate_content call in asyncio.to_thread.
    """
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=1500,
    )

    def _blocking_call():
        # FIX: Pass the prompt string directly. The SDK handles the Content/Part wrapping.
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config,
        )
        return response.text

    return await asyncio.to_thread(_blocking_call)

# ---------------------------------------------------------------------------
# AGENT 1 — BIAS MITIGATION GUARDRAIL
# ---------------------------------------------------------------------------

BIAS_GUARDRAIL_SYSTEM = """
You are a bias-mitigation filter for an AI credit system serving African informal economy workers.

Your job is to:
1. Extract typed trade events from the merchant's message
2. Identify and remove ANY proxy variables that could introduce demographic bias
3. Return ONLY structured JSON — no prose, no markdown, no preamble

PROXY VARIABLES TO REMOVE:
- Location names that signal ethnicity or socioeconomic status (replace with "market area")
- Gender-coded business descriptors ("mama mboga" → "vegetable vendor")
- Tribal/ethnic markers in any language
- Religious time references that signal identity ("after Friday prayers" → "weekly")
- Relative wealth signals ("my mud house" → strip entirely)

TRADE EVENT TYPES:
- revenue: money received from sales
- expense: money paid out for business costs
- inventory_in: stock/goods acquired
- debt_repayment: paying back a loan or credit
- receivable: money owed TO the merchant
- unknown: cannot determine type with confidence

Return this exact JSON structure:
{
  "cleaned_text": "sanitized version of the input",
  "detected_trade_events": [
    {
      "event_type": "revenue|expense|inventory_in|debt_repayment|receivable|unknown",
      "amount_ksh": null or number,
      "frequency_signal": null or "daily|weekly|market_day|irregular",
      "confidence": 0.0-1.0,
      "raw_utterance": "exact fragment from input",
      "bias_flags": ["list of proxy variables removed from this event"]
    }
  ],
  "language": "sw|en|luo|mixed",
  "low_confidence_count": number,
  "overall_confidence": 0.0-1.0,
  "bias_proxy_removed": ["complete list of all proxy variables removed"]
}

CONFIDENCE RULES:
- Set confidence < 0.6 if the amount is ambiguous or inferred
- Set confidence < 0.5 if event_type is uncertain
- Set overall_confidence = average of individual event confidences
- If input has NO extractable trade events, return one event with type "unknown" and confidence 0.0
"""

async def run_bias_mitigation_guardrail(
    raw_message: str,
    merchant_id: str,
    language: str,
    ai_client=None,
) -> SanitizedInput:
    if ai_client is None:
        logger.warning(f"[{merchant_id}] Guardrail running without AI client — using fallback")
        return _guardrail_fallback(raw_message, language)

    prompt = f"""
Merchant message (language hint: {language}):
\"\"\"
{raw_message}
\"\"\"

Extract all trade events and apply bias filtering. Return only the JSON object.
"""

    try:
        raw_response = await _gemini_generate(ai_client, prompt, system=BIAS_GUARDRAIL_SYSTEM)

        # FIX: Bulletproof JSON extraction. Finds the first '{' and the last '}'
        clean = raw_response.strip()
        start = clean.find('{')
        end = clean.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end+1]
        else:
            raise ValueError(f"No JSON object found in response")

        data = json.loads(clean)
        sanitized = SanitizedInput(**data)

        logger.info(
            f"[{merchant_id}] Guardrail: {len(sanitized.detected_trade_events)} events, "
            f"confidence={sanitized.overall_confidence:.2f}, "
            f"bias_removed={sanitized.bias_proxy_removed}"
        )
        return sanitized

    except Exception as e:
        logger.error(f"[{merchant_id}] Guardrail parse error: {e} — using fallback")
        return _guardrail_fallback(raw_message, language)

def _guardrail_fallback(raw_message: str, language: str) -> SanitizedInput:
    return SanitizedInput(
        cleaned_text=raw_message,
        detected_trade_events=[
            TradeEvent(
                event_type="unknown",
                amount_ksh=None,
                confidence=0.1,
                raw_utterance=raw_message[:200],
                bias_flags=[],
            )
        ],
        language=language,
        low_confidence_count=1,
        overall_confidence=0.1,
        bias_proxy_removed=[],
    )

# ---------------------------------------------------------------------------
# AGENT 2 — KIPAJI UNDERWRITER CORE
# ---------------------------------------------------------------------------

UNDERWRITER_SYSTEM = """
You are an AI credit underwriter for Kipaji, serving informal economy MSMEs in East Africa.

You receive:
1. Sanitized trade events (already bias-filtered) from the guardrail agent
2. Merchant's 7-day transaction history from the ledger
3. Computed velocity metrics

YOUR MANDATE:
- Base credit decisions ONLY on trade velocity and consistency — NOT on demographics
- Prioritise access: lean toward approval for borderline cases where velocity signals are positive
- Never penalise merchants for having no formal banking history

CREDIT TIERS (in KSH):
- micro:  500–2,000    (new merchants or low velocity)
- small:  2,001–8,000  (consistent daily traders)
- medium: 8,001–25,000 (high velocity, proven consistency)
- declined: 0          (insufficient signal or very low confidence)

INTEREST RATES:
- micro: 4% monthly, small: 3.5% monthly, medium: 3% monthly
- REPAYMENT: 7 days for micro, 14 days for small, 21 days for medium

Return ONLY this JSON — no prose, no markdown:
{
  "approved": true/false,
  "approved_local": number (KSH),
  "credit_tier": "micro|small|medium|declined",
  "interest_rate_monthly": number,
  "repayment_days": number,
  "velocity_score": number 0-100,
  "consistency_score": number 0-1,
  "confidence_gate_triggered": true/false,
  "decision_reason": "one sentence, internal audit use",
  "audit_trail": ["list of factors considered"],
  "response_message": "merchant-facing message in their language (sw/en/luo), warm and respectful"
}
"""

async def run_kipaji_underwriter_core(
    sanitized: SanitizedInput,
    history: List[Dict[str, Any]],
    profile: Dict[str, Any],
    velocity_metrics: Dict[str, float],
    merchant_id: str,
    ai_client=None,
) -> CreditDecision:
    if sanitized.overall_confidence < 0.65:
        logger.info(f"[{merchant_id}] Confidence gate triggered ({sanitized.overall_confidence:.2f})")
        return _low_confidence_decision(sanitized, merchant_id)

    revenue_events = [
        e for e in sanitized.detected_trade_events
        if e.event_type in ("revenue", "receivable") and e.amount_ksh
    ]

    prompt = f"""
Merchant ID: {merchant_id}
Language: {sanitized.language}

SANITIZED TRADE EVENTS (bias-filtered, revenue only):
{json.dumps([e.model_dump() for e in revenue_events], indent=2)}

7-DAY TRANSACTION HISTORY (last 10 entries):
{json.dumps(history[-10:], indent=2)}

COMPUTED VELOCITY METRICS:
- avg_daily_ksh: {velocity_metrics.get('avg_daily', 0):.2f}
- total_7d_ksh: {velocity_metrics.get('total_7d', 0):.2f}
- consistency_score: {velocity_metrics.get('consistency', 1.0):.2f}

MERCHANT PROFILE:
{json.dumps(profile, indent=2)}

Make a credit decision based solely on trade velocity signals. Return only the JSON object.
"""

    if ai_client is None:
        logger.warning(f"[{merchant_id}] Underwriter running without AI client — using rule-based fallback")
        return _rule_based_decision(revenue_events, velocity_metrics, sanitized.language, merchant_id)

    try:
        raw_response = await _gemini_generate(ai_client, prompt, system=UNDERWRITER_SYSTEM)

        # FIX: Bulletproof JSON extraction
        clean = raw_response.strip()
        start = clean.find('{')
        end = clean.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end+1]
        else:
            raise ValueError(f"No JSON object found in response")

        data = json.loads(clean)
        data["timestamp"] = datetime.utcnow().isoformat()
        decision = CreditDecision(**data)

        logger.info(
            f"[{merchant_id}] Decision: {decision.credit_tier} | "
            f"KSH {decision.approved_local} | "
            f"velocity={decision.velocity_score:.1f} | "
            f"confidence_gate={decision.confidence_gate_triggered}"
        )
        return decision

    except Exception as e:
        logger.error(f"[{merchant_id}] Underwriter parse error: {e} — using rule-based fallback")
        return _rule_based_decision(revenue_events, velocity_metrics, sanitized.language, merchant_id)

# ---------------------------------------------------------------------------
# FALLBACKS & RULE-BASED SCORER
# ---------------------------------------------------------------------------

def _low_confidence_decision(sanitized: SanitizedInput, merchant_id: str) -> CreditDecision:
    lang = sanitized.language
    if lang == "sw":
        msg = "Samahani, hatukuelewa vizuri. Tafadhali tuambie: uliuza nini leo na bei ngapi?"
    elif lang == "luo":
        msg = "Kony, wawacho ne ok wawinjo maber. Nyiswa: ne icho ang'o kawuono kod nengo adi?"
    else:
        msg = "Sorry, we couldn't understand your message clearly. Could you tell us: what did you sell today and for how much?"

    return CreditDecision(
        approved=False,
        approved_local=0,
        credit_tier="declined",
        interest_rate_monthly=0,
        repayment_days=0,
        velocity_score=0,
        consistency_score=0,
        confidence_gate_triggered=True,
        decision_reason=f"Confidence gate: overall_confidence={sanitized.overall_confidence:.2f}",
        audit_trail=["Confidence below threshold 0.65", f"Low confidence events: {sanitized.low_confidence_count}"],
        response_message=msg,
        timestamp=datetime.utcnow().isoformat(),
    )

def _rule_based_decision(
    revenue_events: list,
    velocity_metrics: Dict[str, float],
    language: str,
    merchant_id: str,
) -> CreditDecision:
    avg_daily = velocity_metrics.get("avg_daily", 0)
    consistency = velocity_metrics.get("consistency", 0.5)
    total_7d = velocity_metrics.get("total_7d", 0)

    velocity_score = min(100, (avg_daily / 1000) * 40 + (total_7d / 5000) * 40 + consistency * 20)

    if avg_daily >= 3000 and consistency >= 0.6:
        tier, amount, rate, days = "medium", min(avg_daily * 2.5, 25000), 3.0, 21
    elif avg_daily >= 800 and consistency >= 0.4:
        tier, amount, rate, days = "small", min(avg_daily * 2.0, 8000), 3.5, 14
    elif avg_daily >= 200 or (total_7d > 0 and len(revenue_events) > 0):
        tier, amount, rate, days = "micro", min(max(avg_daily * 1.5, 500), 2000), 4.0, 7
    else:
        tier, amount, rate, days = "declined", 0, 0, 0

    if language == "sw":
        msg = f"Hongera! Umeidhinishiwa mkopo wa KSH {amount:,.0f} kwa siku {days}." if amount > 0 else "Samahani, hatuna taarifa za kutosha za biashara yako bado."
    elif language == "luo":
        msg = f"Ber ahinya! Imiyo KSH {amount:,.0f} mar {days} ndalo." if amount > 0 else "Kony, weche mag ohala mago ok oromo."
    else:
        msg = f"Approved! KSH {amount:,.0f} credit for {days} days." if amount > 0 else "We need a bit more trading history to approve credit."

    return CreditDecision(
        approved=amount > 0,
        approved_local=round(amount, 2),
        credit_tier=tier,
        interest_rate_monthly=rate,
        repayment_days=days,
        velocity_score=round(velocity_score, 1),
        consistency_score=round(consistency, 2),
        confidence_gate_triggered=False,
        decision_reason=f"Rule-based fallback: avg_daily={avg_daily:.0f} KSH, consistency={consistency:.2f}",
        audit_trail=[
            f"avg_daily_ksh={avg_daily:.2f}",
            f"total_7d_ksh={total_7d:.2f}",
            f"consistency={consistency:.2f}",
            f"revenue_events_count={len(revenue_events)}",
            "LLM unavailable — rule-based scoring used",
        ],
        response_message=msg,
        timestamp=datetime.utcnow().isoformat(),
    )