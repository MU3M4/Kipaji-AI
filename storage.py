"""
storage.py — Kipaji-AI data layer (In-Memory Edition)
Handles merchant ledger and USSD session state entirely in memory.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("KipajiStorage")

# In-memory stores
_MERCHANT_LEDGER: Dict[str, List[Dict[str, Any]]] = {}
_MERCHANT_PROFILES: Dict[str, Dict[str, Any]] = {}
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}

async def get_merchant_data(merchant_id: str, db=None) -> Dict[str, Any]:
    """Returns {"history": [...], "profile": {...}} for a merchant."""
    return {
        "history": _MERCHANT_LEDGER.get(merchant_id, []),
        "profile": _MERCHANT_PROFILES.get(merchant_id, {}),
    }

async def save_trade_interaction(
    merchant_id: str,
    extracted_trade_amount: float,
    event_type: str,
    decision_dict: Dict[str, Any],
    bias_removed: List[str],
    language: str,
    db=None,
) -> None:
    """Persists a trade event to the merchant's history ledger."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": event_type,
        "amount_local": extracted_trade_amount,
        "language": language,
        "bias_removed": bias_removed,
        "credit_decision": {
            "tier": decision_dict.get("credit_tier"),
            "approved_local": decision_dict.get("approved_local"),
        },
    }
    
    if merchant_id not in _MERCHANT_LEDGER:
        _MERCHANT_LEDGER[merchant_id] = []
    _MERCHANT_LEDGER[merchant_id].append(entry)
    
    # Cap at 270 entries to prevent memory leaks
    if len(_MERCHANT_LEDGER[merchant_id]) > 270:
        _MERCHANT_LEDGER[merchant_id] = _MERCHANT_LEDGER[merchant_id][-270:]
        
    logger.info(f"[{merchant_id}] Trade event saved in-memory: {event_type} KSH {extracted_trade_amount}")

async def get_ussd_session(session_id: str, redis_client=None) -> Optional[Dict[str, Any]]:
    return _SESSION_STORE.get(session_id)

async def save_ussd_session(session_id: str, state: Dict[str, Any], redis_client=None) -> None:
    state["last_updated"] = datetime.utcnow().isoformat()
    _SESSION_STORE[session_id] = state

async def clear_ussd_session(session_id: str, redis_client=None) -> None:
    _SESSION_STORE.pop(session_id, None)