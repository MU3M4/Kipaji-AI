"""
test_kipaji.py — Unit + integration tests for Kipaji-AI

Run with: python -m pytest test_kipaji.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Tests for main.py helpers
# ─────────────────────────────────────────────────────────────────────────────

from main import detect_language, calculate_velocity_metrics


class TestLanguageDetection:
    def test_swahili_markers(self):
        assert detect_language("Nimeuza mahindi leo bei 50") == "sw"

    def test_dholuo_markers(self):
        assert detect_language("Aora pesa mar ohala") == "luo"

    def test_english_fallback(self):
        result = detect_language("I sold tomatoes today for 200")
        assert result in ("en", "sw")   # langdetect may vary

    def test_sheng_treated_as_swahili(self):
        assert detect_language("Niko na chapaa leo nlifanya biashara") == "sw"

    def test_empty_string_does_not_crash(self):
        result = detect_language("")
        assert result in ("sw", "en", "luo")

    def test_mixed_code_swahili_wins(self):
        assert detect_language("nimeuza ksh 500 leo shs 200 faida") == "sw"


class TestVelocityMetrics:
    def test_empty_history_returns_zeros(self):
        result = calculate_velocity_metrics([])
        assert result["avg_daily"] == 0.0
        assert result["total_7d"] == 0.0

    def test_avg_daily_divides_by_7_not_count(self):
        """
        BUG FIX TEST: avg_daily must divide by 7 (the window), not by transaction count.
        A high-frequency trader with 10 transactions should NOT be penalised.
        """
        history = [
            {"type": "revenue", "amount_local": 1000.0} for _ in range(3)
        ]
        result = calculate_velocity_metrics(history)
        # 3 transactions totalling 3000 KSH over 7 days = 428.57 avg_daily
        assert result["total_7d"] == 3000.0
        assert result["avg_daily"] == pytest.approx(3000.0 / 7, rel=0.01)

    def test_high_frequency_trader_not_penalised(self):
        """7 transactions vs 3 transactions with same total — avg_daily should be identical."""
        history_3 = [{"type": "revenue", "amount_local": 1000.0} for _ in range(3)]
        history_7 = [{"type": "revenue", "amount_local": 1000.0 / 7 * 3} for _ in range(7)]

        result_3 = calculate_velocity_metrics(history_3)
        result_7 = calculate_velocity_metrics(history_7)
        # Both have ~3000 KSH total, so avg_daily should be the same
        assert abs(result_3["avg_daily"] - result_7["avg_daily"]) < 1.0

    def test_filters_non_revenue_events(self):
        history = [
            {"type": "revenue", "amount_local": 500.0},
            {"type": "expense", "amount_local": 9999.0},   # should be ignored
            {"type": "debt_repayment", "amount_local": 9999.0},
        ]
        result = calculate_velocity_metrics(history)
        assert result["total_7d"] == 500.0

    def test_filters_zero_amounts(self):
        history = [
            {"type": "revenue", "amount_local": 0},
            {"type": "revenue", "amount_local": 1000.0},
        ]
        result = calculate_velocity_metrics(history)
        assert result["total_7d"] == 1000.0

    def test_consistency_single_entry_is_neutral(self):
        history = [{"type": "revenue", "amount_local": 1000.0}]
        result = calculate_velocity_metrics(history)
        assert result["consistency"] == 0.5   # neutral, not 1.0

    def test_consistency_even_amounts_is_high(self):
        history = [{"type": "revenue", "amount_local": 500.0} for _ in range(7)]
        result = calculate_velocity_metrics(history)
        assert result["consistency"] > 0.9

    def test_transaction_count_is_returned(self):
        history = [{"type": "revenue", "amount_local": 100.0} for _ in range(5)]
        result = calculate_velocity_metrics(history)
        assert result["transaction_count"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Tests for agents.py
# ─────────────────────────────────────────────────────────────────────────────

from agents import (
    run_bias_mitigation_guardrail,
    run_kipaji_underwriter_core,
    SanitizedInput,
    TradeEvent,
    CreditDecision,
    _guardrail_fallback,
    _low_confidence_decision,
    _rule_based_decision,
)


class TestGuardrailFallback:
    def test_fallback_returns_valid_sanitized_input(self):
        result = _guardrail_fallback("test message", "sw")
        assert isinstance(result, SanitizedInput)
        assert result.overall_confidence == 0.1
        assert len(result.detected_trade_events) == 1
        assert result.detected_trade_events[0].event_type == "unknown"

    def test_fallback_does_not_crash_on_long_message(self):
        long_msg = "a " * 500
        result = _guardrail_fallback(long_msg, "en")
        assert result.detected_trade_events[0].raw_utterance  # truncated to 200


class TestLowConfidenceDecision:
    def test_returns_declined_tier(self):
        sanitized = _guardrail_fallback("unclear", "sw")
        decision = _low_confidence_decision(sanitized, "merchant_001")
        assert decision.credit_tier == "declined"
        assert decision.approved is False
        assert decision.confidence_gate_triggered is True

    def test_swahili_message(self):
        sanitized = _guardrail_fallback("unclear", "sw")
        decision = _low_confidence_decision(sanitized, "m1")
        assert "Samahani" in decision.response_message or "uliuza" in decision.response_message.lower()

    def test_luo_message(self):
        sanitized = _guardrail_fallback("unclear", "luo")
        decision = _low_confidence_decision(sanitized, "m2")
        assert "Kony" in decision.response_message or "wawacho" in decision.response_message


class TestRuleBasedDecision:
    def _make_events(self, amount: float, confidence: float = 0.8):
        return [TradeEvent(
            event_type="revenue",
            amount_ksh=amount,
            confidence=confidence,
            raw_utterance="test",
        )]

    def test_high_velocity_gets_medium_tier(self):
        metrics = {"avg_daily": 4000.0, "total_7d": 28000.0, "consistency": 0.8}
        decision = _rule_based_decision(self._make_events(4000), metrics, "sw", "m1")
        assert decision.credit_tier == "medium"
        assert decision.approved is True
        assert decision.approved_local <= 25000

    def test_medium_velocity_gets_small_tier(self):
        metrics = {"avg_daily": 1000.0, "total_7d": 7000.0, "consistency": 0.5}
        decision = _rule_based_decision(self._make_events(1000), metrics, "sw", "m1")
        assert decision.credit_tier == "small"

    def test_low_velocity_gets_micro_tier(self):
        metrics = {"avg_daily": 300.0, "total_7d": 2100.0, "consistency": 0.4}
        decision = _rule_based_decision(self._make_events(300), metrics, "sw", "m1")
        assert decision.credit_tier == "micro"

    def test_zero_velocity_gets_declined(self):
        metrics = {"avg_daily": 0.0, "total_7d": 0.0, "consistency": 0.0}
        decision = _rule_based_decision([], metrics, "sw", "m1")
        assert decision.credit_tier == "declined"
        assert decision.approved is False

    def test_approved_local_does_not_exceed_tier_ceiling(self):
        metrics = {"avg_daily": 50000.0, "total_7d": 350000.0, "consistency": 1.0}
        decision = _rule_based_decision(self._make_events(50000), metrics, "en", "m1")
        assert decision.approved_local <= 25000   # medium ceiling

    def test_audit_trail_is_populated(self):
        metrics = {"avg_daily": 1000.0, "total_7d": 7000.0, "consistency": 0.6}
        decision = _rule_based_decision(self._make_events(1000), metrics, "sw", "m1")
        assert len(decision.audit_trail) > 0
        assert any("avg_daily" in item for item in decision.audit_trail)

    def test_english_response_message(self):
        metrics = {"avg_daily": 1000.0, "total_7d": 7000.0, "consistency": 0.5}
        decision = _rule_based_decision(self._make_events(1000), metrics, "en", "m1")
        assert "KSH" in decision.response_message or "Approved" in decision.response_message


class TestGuardrailAsync:
    """Tests for run_bias_mitigation_guardrail — uses no AI client (fallback path)."""

    def test_no_client_returns_fallback(self):
        result = asyncio.run(
            run_bias_mitigation_guardrail("nimeuza mahindi 500", "m1", "sw", ai_client=None)
        )
        assert isinstance(result, SanitizedInput)
        assert result.overall_confidence == 0.1   # fallback confidence

    def test_valid_sanitized_input_structure(self):
        result = asyncio.run(
            run_bias_mitigation_guardrail("sold tomatoes for 200 today", "m1", "en", ai_client=None)
        )
        assert hasattr(result, "detected_trade_events")
        assert hasattr(result, "bias_proxy_removed")
        assert hasattr(result, "overall_confidence")


class TestUnderwriterAsync:
    """Tests for run_kipaji_underwriter_core — uses no AI client (rule-based fallback)."""

    def _make_sanitized(self, confidence: float = 0.8) -> SanitizedInput:
        return SanitizedInput(
            cleaned_text="sold maize 500 ksh",
            detected_trade_events=[
                TradeEvent(
                    event_type="revenue",
                    amount_ksh=500.0,
                    frequency_signal="daily",
                    confidence=confidence,
                    raw_utterance="sold maize 500 ksh",
                )
            ],
            language="sw",
            low_confidence_count=0,
            overall_confidence=confidence,
            bias_proxy_removed=[],
        )

    def test_confidence_gate_triggers_below_065(self):
        sanitized = self._make_sanitized(confidence=0.4)
        decision = asyncio.run(run_kipaji_underwriter_core(
            sanitized=sanitized,
            history=[],
            profile={},
            velocity_metrics={"avg_daily": 500.0, "total_7d": 3500.0, "consistency": 0.6},
            merchant_id="m1",
            ai_client=None,
        ))
        assert decision.confidence_gate_triggered is True
        assert decision.credit_tier == "declined"

    def test_above_threshold_produces_decision(self):
        sanitized = self._make_sanitized(confidence=0.9)
        history = [{"type": "revenue", "amount_local": 500.0} for _ in range(5)]
        decision = asyncio.run(run_kipaji_underwriter_core(
            sanitized=sanitized,
            history=history,
            profile={},
            velocity_metrics={"avg_daily": 300.0, "total_7d": 2100.0, "consistency": 0.7},
            merchant_id="m1",
            ai_client=None,
        ))
        assert isinstance(decision, CreditDecision)
        assert decision.credit_tier in ("micro", "small", "medium", "declined")
        assert decision.timestamp  # populated


# ─────────────────────────────────────────────────────────────────────────────
# Tests for storage.py
# ─────────────────────────────────────────────────────────────────────────────

from storage import (
    get_merchant_data,
    save_trade_interaction,
    get_ussd_session,
    save_ussd_session,
    clear_ussd_session,
    _MERCHANT_LEDGER,
    _SESSION_STORE,
)


class TestStorageInMemory:
    def setup_method(self):
        _MERCHANT_LEDGER.clear()
        _SESSION_STORE.clear()

    def test_get_merchant_data_empty_returns_empty(self):
        result = asyncio.run(get_merchant_data("new_merchant", db=None))
        assert result == {"history": [], "profile": {}}

    def test_save_stores_trade_amount_not_credit(self):
        """
        BUG FIX TEST: save_trade_interaction must store extracted_trade_amount,
        not approved_local from the credit decision.
        """
        asyncio.run(save_trade_interaction(
            merchant_id="m1",
            extracted_trade_amount=800.0,          # real trade amount
            event_type="revenue",
            decision_dict={"credit_tier": "micro", "approved_local": 1000.0},  # different!
            bias_removed=[],
            language="sw",
            db=None,
        ))
        data = asyncio.run(get_merchant_data("m1", db=None))
        entry = data["history"][0]
        assert entry["amount_local"] == 800.0     # stored trade amount, not 1000 credit
        assert entry["type"] == "revenue"

    def test_ussd_session_roundtrip(self):
        asyncio.run(save_ussd_session("sess_001", {"step": 1, "merchant": "m1"}, redis_client=None))
        result = asyncio.run(get_ussd_session("sess_001", redis_client=None))
        assert result["step"] == 1

    def test_ussd_session_clear(self):
        asyncio.run(save_ussd_session("sess_002", {"step": 1}, redis_client=None))
        asyncio.run(clear_ussd_session("sess_002", redis_client=None))
        result = asyncio.run(get_ussd_session("sess_002", redis_client=None))
        assert result is None

    def test_merchant_ledger_capped_at_270(self):
        for i in range(300):
            asyncio.run(save_trade_interaction("m2", 100.0, "revenue", {}, [], "sw", None))
        data = asyncio.run(get_merchant_data("m2", db=None))
        assert len(data["history"]) == 270


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI integration test
# ─────────────────────────────────────────────────────────────────────────────

from fastapi.testclient import TestClient
from main import app


class TestAPIEndpoints:
    def setup_method(self):
        _MERCHANT_LEDGER.clear()
        _SESSION_STORE.clear()

    def test_root_returns_online(self):
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ONLINE"

    def test_health_endpoint_exists(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "healthy" in resp.json()

    def test_gateway_processes_whatsapp_message(self):
        client = TestClient(app)
        resp = client.post("/api/v1/gateway", json={
            "merchant_id": "test_merchant_001",
            "phone_number": "+254712345678",
            "channel": "whatsapp",
            "message_body": "Nimeuza mahindi leo kwa shilingi 500",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert "credit_decision" in data
        assert "velocity_metrics" in data
        assert data["language_detected"] == "sw"

    def test_gateway_invalid_channel_rejected(self):
        client = TestClient(app)
        resp = client.post("/api/v1/gateway", json={
            "merchant_id": "m1",
            "phone_number": "+254712345678",
            "channel": "telegram",        # not allowed
            "message_body": "test",
        })
        assert resp.status_code == 422

    def test_gateway_invalid_phone_rejected(self):
        client = TestClient(app)
        resp = client.post("/api/v1/gateway", json={
            "merchant_id": "m1",
            "phone_number": "not-a-phone",
            "channel": "whatsapp",
            "message_body": "test",
        })
        assert resp.status_code == 422

    def test_merchant_history_endpoint(self):
        client = TestClient(app)
        resp = client.get("/api/v1/merchant/any_merchant/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "velocity_metrics" in data
        assert "trade_history_count" in data


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])
