import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("KipajiStorage")

# --- IN-MEMORY FALLBACKS ---
_MERCHANT_LEDGER: Dict[str, List[Dict[str, Any]]] = {}
_MERCHANT_PROFILES: Dict[str, Dict[str, Any]] = {}
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# --- MERCHANT LEDGER ---
async def get_merchant_data(merchant_id: str, db=None) -> Dict[str, Any]:
    if db is not None:
        try:
            merchant_ref = db.collection("merchants").document(merchant_id)
            def _read():
                doc = merchant_ref.get()
                if not doc.exists: return {"history": [], "profile": {}}
                data = doc.to_dict()
                return {"history": data.get("trade_history", []), "profile": data.get("profile", {})}
            return await asyncio.to_thread(_read)
        except Exception as e:
            logger.error(f"[{merchant_id}] Firestore read error: {e}")
            
    return {"history": _MERCHANT_LEDGER.get(merchant_id, []), "profile": _MERCHANT_PROFILES.get(merchant_id, {})}

async def save_trade_interaction(merchant_id: str, extracted_trade_amount: float, event_type: str, decision_dict: Dict[str, Any], bias_removed: List[str], language: str, db=None) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(), "type": event_type,
        "amount_local": extracted_trade_amount, "language": language, "bias_removed": bias_removed,
        "credit_decision": {"tier": decision_dict.get("credit_tier"), "approved_local": decision_dict.get("approved_local")}
    }
    
    if db is not None:
        try:
            def _write():
                merchant_ref = db.collection("merchants").document(merchant_id)
                merchant_ref.set({"trade_history": db.field_path_transforms.ArrayUnion([entry])}, merge=True)
            await asyncio.to_thread(_write)
            return
        except Exception as e:
            logger.error(f"[{merchant_id}] Firestore write error: {e}")

    if merchant_id not in _MERCHANT_LEDGER: _MERCHANT_LEDGER[merchant_id] = []
    _MERCHANT_LEDGER[merchant_id].append(entry)
    if len(_MERCHANT_LEDGER[merchant_id]) > 270: _MERCHANT_LEDGER[merchant_id] = _MERCHANT_LEDGER[merchant_id][-270:]

# --- USSD SESSION STATE ---
async def get_ussd_session(session_id: str, redis_client=None) -> Optional[Dict[str, Any]]:
    if redis_client is not None:
        try:
            def _read():
                raw = redis_client.get(f"ussd:{session_id}")
                return json.loads(raw) if raw else None
            return await asyncio.to_thread(_read)
        except Exception as e:
            logger.error(f"Redis read error: {e}")
    return _SESSION_STORE.get(session_id)

async def save_ussd_session(session_id: str, state: Dict[str, Any], redis_client=None) -> None:
    state["last_updated"] = datetime.utcnow().isoformat()
    if redis_client is not None:
        try:
            def _write(): redis_client.setex(f"ussd:{session_id}", 180, json.dumps(state))
            await asyncio.to_thread(_write)
            return
        except Exception as e:
            logger.error(f"Redis write error: {e}")
    _SESSION_STORE[session_id] = state

async def clear_ussd_session(session_id: str, redis_client=None) -> None:
    if redis_client is not None:
        try: await asyncio.to_thread(redis_client.delete, f"ussd:{session_id}")
        except Exception as e: logger.error(f"Redis delete error: {e}")
    _SESSION_STORE.pop(session_id, None)