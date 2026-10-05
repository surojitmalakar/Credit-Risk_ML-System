"""Tests for the optional free LLM endpoint and copilot fallback."""

from unittest.mock import patch

import pandas as pd
import pytest

from msme_ews.copilot import (
    generate_credit_copilot_answer,
    generate_dataset_copilot_answer,
)
from msme_ews.llm import (
    LLMSettings,
    build_messages,
    chat_completion,
    context_prompt,
    dataset_context_prompt,
    load_settings,
)


def test_settings_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = load_settings({})
    assert settings.enabled is False
    assert settings.label == "rule-based engine"


def test_settings_load_from_environment(monkeypatch):
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    settings = load_settings()
    assert settings.enabled is True
    assert settings.model == "llama3.2"
    assert "local model" in settings.label


def test_settings_load_from_explicit_environ():
    settings = load_settings({
        "LLM_API_BASE": "https://api.groq.com/openai/v1",
        "LLM_API_KEY": "gsk_test",
        "LLM_MODEL": "llama-3.3-70b-versatile",
        "LLM_TIMEOUT": "5.5",
    })
    assert settings.api_base == "https://api.groq.com/openai/v1"
    assert settings.api_key == "gsk_test"
    assert settings.model == "llama-3.3-70b-versatile"
    assert settings.timeout == 5.5
    assert "groq.com" in settings.label


def test_settings_invalid_timeout_falls_back():
    settings = load_settings({"LLM_API_BASE": "http://x", "LLM_TIMEOUT": "abc"})
    assert settings.timeout == 30.0  # DEFAULT_TIMEOUT


def test_context_prompt_renders_observed_figures():
    context = {
        "company": "C-1",
        "risk_category": "Elevated",
        "revenue": 123456.789,
        "current_ratio": None,
        "warnings": ["Falling liquidity", "Increasing leverage"],
        "top_risk": [{"feature": "Debt_to_Assets", "contribution": 0.4}],
        "prior_period": {"period": "2024", "revenue": 100000.0},
    }
    prompt = context_prompt(context)
    assert "company: C-1" in prompt
    assert "revenue: 123457" in prompt
    assert "current_ratio: not available" in prompt
    assert "warnings: Falling liquidity, Increasing leverage" in prompt
    assert "top_risk: Debt_to_Assets" in prompt
    assert "prior_period: period 2024; revenue 100000" in prompt


def test_dataset_context_prompt_renders_metadata():
    analysis = {
        "data": None,
        "numeric_columns": ["Revenue", "EBITDA"],
        "categorical_columns": ["sector"],
        "missing_percent": 2.5,
        "duplicate_count": 3,
        "anomaly_count": 12,
        "analysis_mode": "general",
        "executive_summary": "Profile summary",
        "targets": ["distress_label"],
        "recommendations": ["Add more periods", "Check coverage"],
    }
    prompt = dataset_context_prompt(analysis)
    assert "dataset rows: 0" in prompt
    assert "numeric columns: Revenue, EBITDA" in prompt
    assert "missing cells: 2.5%" in prompt
    assert "detected targets: distress_label" in prompt
    assert "recommendations: Add more periods | Check coverage" in prompt


def test_build_messages_structure():
    messages = build_messages("Why risky?", "company: C-1\nrisk: Elevated", ["earlier question"])
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Earlier question: earlier question"
    assert "Credit context" in messages[2]["content"]
    assert "Question: Why risky?" in messages[2]["content"]


def test_chat_completion_requires_endpoint():
    with pytest.raises(RuntimeError, match="No LLM endpoint"):
        chat_completion([{"role": "user", "content": "hi"}], LLMSettings())


def test_chat_completion_unreachable_endpoint():
    settings = LLMSettings(api_base="http://127.0.0.1:1/v1", timeout=1.0)
    with pytest.raises(RuntimeError, match="unreachable"):
        chat_completion([{"role": "user", "content": "hi"}], settings)


def test_credit_copilot_falls_back_when_llm_fails():
    context = {
        "company": "C-1",
        "period": "2025",
        "revenue": 1000.0,
        "revenue_growth": None,
        "profit": 100.0,
        "current_ratio": 0.8,
        "debt_to_assets": 0.7,
        "ebitda_margin": -0.1,
        "interest_coverage": None,
        "cash_flow_operations": -50.0,
        "default_probability": None,
        "risk_index": 62.0,
        "health_score": 38.0,
        "risk_category": "High Risk",
        "assessment_method": "Rule-based index",
        "coverage_label": "9/14 core fields (64%)",
        "coverage_percent": 64.0,
        "warnings": ["Falling liquidity"],
        "top_risk": [],
        "protective": [],
        "history_available": False,
        "prior_period": None,
    }
    broken = LLMSettings(api_base="http://127.0.0.1:1/v1", timeout=1.0)
    answer, source = generate_credit_copilot_answer(
        "Summarize the financial health", context, llm=broken
    )
    assert source == "rule-based"
    assert "rule-based risk index" in answer


def test_credit_copilot_uses_llm_when_available():
    context = {"company": "C-1", "risk_category": "Elevated", "warnings": []}
    fake = LLMSettings(api_base="http://fake/v1")
    with patch("msme_ews.copilot.chat_completion", return_value="LLM answer text"):
        answer, source = generate_credit_copilot_answer(
            "Why is C-1 elevated?", context, llm=fake
        )
    assert source == "llm"
    assert answer == "LLM answer text"


def test_dataset_copilot_falls_back_when_llm_fails():
    data = pd.DataFrame({"Revenue": [100.0, 200.0], "sector": ["a", "b"]})
    analysis = {
        "data": data,
        "numeric_columns": ["Revenue"],
        "categorical_columns": ["sector"],
        "missing_percent": 0.0,
        "duplicate_count": 0,
        "anomaly_count": 0,
        "anomaly_rows_scored": 2,
        "analysis_mode": "general",
        "executive_summary": "Summary text",
        "recommendations": [],
        "profile": pd.DataFrame(),
        "trends": pd.DataFrame(),
        "correlations": pd.DataFrame(),
        "warning_details": [],
        "model": {},
    }
    broken = LLMSettings(api_base="http://127.0.0.1:1/v1", timeout=1.0)
    answer, source = generate_dataset_copilot_answer(
        "What about Revenue?", analysis, llm=broken
    )
    assert source == "rule-based"
    assert "Revenue" in answer


def test_dataset_copilot_uses_llm_when_available():
    analysis = {"executive_summary": "Summary text"}
    fake = LLMSettings(api_base="http://fake/v1")
    with patch("msme_ews.copilot.chat_completion", return_value="LLM dataset answer"):
        answer, source = generate_dataset_copilot_answer(
            "What are the main warnings?", analysis, llm=fake
        )
    assert source == "llm"
    assert answer == "LLM dataset answer"
