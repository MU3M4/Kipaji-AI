"""
storage.py — Kipaji-AI data layer

Handles:
- Merchant ledger reads/writes (Firestore with in-memory fallback)
- USSD session state (Redis with in-memory fallback)
- Structured trade event persistence (stores extracted amounts, not credit outputs)
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("KipajiStorage")

# ---------------------------------------------------------------------------
# IN-MEMORY FALLBACKS (used when Firestore / Redis are unavailable)
# ---------------------------------------------------------------------------
_MERCHANT_LEDGER: Dict[str, List[Dict[str, Any]]] = {}
_MERCHANT_PROFILES: Dict[str, Dict[str, Any]] = {}
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}   # keyed by ussd_session_id


# ---------------------------------------------------------------------------
# MERCHANT LEDGER (Firestore)
# ---------------------------------------------------------------------------

async def get_merchant_data(merchant_id: str, db=None) -> Dict[str, Any]:
    """
    Returns {"history": [...], "profile": {...}} for a merchant.
    history contains real trade events — NOT credit decisions.
    """
    if db is not None:
        try:
            merchant_ref = db.collection("merchants").document(merchant_id)

            def _read():
                doc = merchant_ref.get()
                if not doc.exists:
                    return {"history": [], "profile": {}}
                data = doc.to_dict()
                return {
                    "history": data.get("trade_history", []),
                    "profile": data.get("profile", {}),
                }

            return await asyncio.to_thread(_read)

        except Exception as e:
            logger.error(f"[{merchant_id}] Firestore read error: {e} — using in-memory fallback")

    # In-memory fallback
    return {
        "history": _MERCHANT_LEDGER.get(merchant_id, []),
        "profile": _MERCHANT_PROFILES.get(merchant_id, {}),
    }


async def save_trade_interaction(
    merchant_id: str,
    extracted_trade_amount: float,       # from NLP agent, NOT from credit decision
    event_type: str,
    decision_dict: Dict[str, Any],
    bias_removed: List[str],
    language: str,
    db=None,
) -> None:
    """
    Persists a trade event to the merchant's history ledger.
    Critically: stores the EXTRACTED TRADE AMOUNT, not the approved credit amount.
    This is what feeds back into velocity calculations on future requests.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": event_type,                          # "revenue", "expense", etc.
        "amount_local": extracted_trade_amount,      # real trade amount in KSH
        "language": language,
        "bias_removed": bias_removed,
        "credit_decision": {
            "tier": decision_dict.get("credit_tier"),
            "approved_local": decision_dict.get("approved_local"),
            "velocity_score": decision_dict.get("velocity_score"),
        },
    }

    if db is not None:
        try:
            def _write():
                merchant_ref = db.collection("merchants").document(merchant_id)
                merchant_ref.set(
                    {"trade_history": db.field_path_transforms.ArrayUnion([entry])},
                    merge=True,
                )

            await asyncio.to_thread(_write)
            logger.info(f"[{merchant_id}] Trade event saved to Firestore: {event_type} KSH {extracted_trade_amount}")
            return
        except Exception as e:
            logger.error(f"[{merchant_id}] Firestore write error: {e} — using in-memory fallback")

    # In-memory fallback
    if merchant_id not in _MERCHANT_LEDGER:
        _MERCHANT_LEDGER[merchant_id] = []
    _MERCHANT_LEDGER[merchant_id].append(entry)
    # Keep last 90 days of entries (cap at 270 entries for in-memory)
    if len(_MERCHANT_LEDGER[merchant_id]) > 270:
        _MERCHANT_LEDGER[merchant_id] = _MERCHANT_LEDGER[merchant_id][-270:]


# ---------------------------------------------------------------------------
# USSD SESSION STATE (Redis with in-memory fallback)
# ---------------------------------------------------------------------------

USSD_SESSION_TTL = 180   # seconds — mirrors Safaricom USSD timeout


async def get_ussd_session(session_id: str, redis_client=None) -> Optional[Dict[str, Any]]:
    """Retrieve a USSD session state by session_id. Returns None if not found."""
    if redis_client is not None:
        try:
            def _read():
                raw = redis_client.get(f"ussd:{session_id}")
                return json.loads(raw) if raw else None
            return await asyncio.to_thread(_read)
        except Exception as e:
            logger.error(f"Redis read error for session {session_id}: {e}")

    return _SESSION_STORE.get(session_id)


async def save_ussd_session(
    session_id: str,
    state: Dict[str, Any],
    redis_client=None,
) -> None:
    """Persist USSD session state with TTL matching the network timeout."""
    state["last_updated"] = datetime.utcnow().isoformat()

    if redis_client is not None:
        try:
            def _write():
                redis_client.setex(
                    f"ussd:{session_id}",
                    USSD_SESSION_TTL,
                    json.dumps(state),
                )
            await asyncio.to_thread(_write)
            return
        except Exception as e:
            logger.error(f"Redis write error for session {session_id}: {e}")

    # In-memory fallback (no TTL enforcement — acceptable for dev/demo)
    _SESSION_STORE[session_id] = state


async def clear_ussd_session(session_id: str, redis_client=None) -> None:
    """Clear a session after completion or timeout."""
    if redis_client is not None:
        try:
            await asyncio.to_thread(redis_client.delete, f"ussd:{session_id}")
        except Exception as e:
            logger.error(f"Redis delete error for session {session_id}: {e}")

    _SESSION_STORE.pop(session_id, None)
