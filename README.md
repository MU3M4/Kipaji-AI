# Kipaji AI — Cash-Velocity Credit Engine

**AI-native short-term credit for informal economy MSMEs in East Africa.**

Kipaji analyses conversational trade messages (WhatsApp, USSD, SMS) to extract cash-velocity signals and issue micro-credit decisions — with active bias mitigation built into the agent pipeline.

---

## Architecture

```
Input channels          Agent pipeline              Output
──────────────          ──────────────              ──────
WhatsApp ──┐            Agent 1: Bias guardrail     Credit decision (KSH)
SMS      ──┼──► FastAPI ────────────────────────►   Merchant-language response
USSD     ──┘   gateway  Agent 2: Underwriter        Audit log (WebSocket)
                │        (confidence gate)           Firestore ledger
                │
                └── Redis session state (USSD continuity)
```

### Key design decisions

- **Typed inter-agent contract** — `TradeEvent` schema enforced at agent boundary. No free-form LLM-to-LLM text passing.
- **Bias guardrail is inline, not post-hoc** — demographic proxies are stripped *before* the scorer sees the data.
- **Confidence gate** — if extraction confidence < 0.65, routes to clarification rather than hallucinating a decision.
- **Gemini calls in `asyncio.to_thread`** — blocking SDK never holds the event loop.
- **WebSocket telemetry is fire-and-forget** — never on the credit decision critical path.
- **Deterministic fallback** — rule-based scorer activates if Gemini is unavailable. Zero single point of failure.

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/MU3M4/Kipaji-AI.git
cd Kipaji-AI
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and GOOGLE_CLOUD_PROJECT

# 3. Run
python main.py

# 4. Test
python -m pytest test_kipaji.py -v
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/gateway` | WhatsApp / SMS / API credit requests |
| POST | `/api/v1/ussd` | USSD gateway (Africa's Talking / Safaricom format) |
| GET | `/api/v1/merchant/{id}/history` | Merchant trade history + velocity metrics |
| GET | `/health` | Service health check |
| WS | `/ws/telemetry` | Real-time audit event stream |

---

## Example: WhatsApp request

```bash
curl -X POST http://localhost:8000/api/v1/gateway \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "merchant_001",
    "phone_number": "+254712345678",
    "channel": "whatsapp",
    "message_body": "Nimeuza mahindi leo kwa shilingi 500 na nyanya 300"
  }'
```

Response:
```json
{
  "status": "SUCCESS",
  "language_detected": "sw",
  "credit_decision": {
    "approved": true,
    "approved_local": 1500.0,
    "credit_tier": "micro",
    "interest_rate_monthly": 4.0,
    "repayment_days": 7,
    "velocity_score": 45.2,
    "confidence_gate_triggered": false,
    "response_message": "Hongera! Umeidhinishiwa mkopo wa KSH 1,500 kwa siku 7."
  },
  "velocity_metrics": {
    "avg_daily": 114.29,
    "total_7d": 800.0,
    "consistency": 0.5
  }
}
```

---

## Bias mitigation — how it works

The bias guardrail agent (Agent 1) runs on every message before scoring:

1. Detects and strips demographic proxy variables (location names that signal ethnicity, gender-coded descriptors, tribal markers)
2. Logs all removed proxies to the audit trail
3. Returns typed `TradeEvent` objects — revenue signals only, separated from expenses and debt

The underwriter (Agent 2) never sees raw text. It only scores the typed, sanitized events.

Every decision includes an `audit_trail` array documenting exactly which signals were used and which were stripped — enabling post-hoc fairness audits.

---

## XPRIZE judging evidence

The `/ws/telemetry` WebSocket streams structured decision logs in real time:

```json
{
  "event": "credit_decision",
  "merchant_id": "merchant_001",
  "timestamp": "2026-06-06T10:23:01.123Z",
  "credit_tier": "small",
  "approved_local": 3500,
  "velocity_score": 62.4,
  "bias_proxy_removed": ["mama mboga → vegetable vendor"],
  "audit_trail": [
    "avg_daily_ksh=700.00",
    "consistency=0.74",
    "revenue_events_count=3"
  ]
}
```

This stream constitutes the "agent execution logs" and "AI running in production" evidence required for submission.
