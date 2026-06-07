"""
storage.py — Kipaji-AI data layer (In-Memory Edition)
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("KipajiStorage")

_MERCHANT_LEDGER: Dict[str, List[Dict[str, Any]]] = {}
_MERCHANT_PROFILES: Dict[str, Dict[str, Any]] = {}

async def get_merchant_data(merchant_id: str, db=None) -> Dict[str, Any]:
    return {
        "history": _MERCHANT_LEDGER.get(merchant_id, []),
        "profile": _MERCHANT_PROFILES.get(merchant_id, {}),
    }

async def save_trade_interaction(
    merchant_id: str, extracted_trade_amount: float, event_type: str,
    decision_dict: Dict[str, Any], bias_removed: List[str], language: str, db=None,
) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(), "type": event_type,
        "amount_local": extracted_trade_amount, "language": language,
        "bias_removed": bias_removed,
        "credit_decision": {"tier": decision_dict.get("credit_tier"), "approved_local": decision_dict.get("approved_local")},
    }
    if merchant_id not in _MERCHANT_LEDGER: _MERCHANT_LEDGER[merchant_id] = []
    _MERCHANT_LEDGER[merchant_id].append(entry)
    if len(_MERCHANT_LEDGER[merchant_id]) > 270: _MERCHANT_LEDGER[merchant_id] = _MERCHANT_LEDGER[merchant_id][-270:]
    logger.info(f"[{merchant_id}] Trade event saved in-memory: {event_type} KSH {extracted_trade_amount}")