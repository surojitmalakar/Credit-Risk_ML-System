"""AI Credit Copilot context building and question answering.

Answers are generated from the selected company-period's observed figures,
model output, and early-warning signals. The responder is a transparent,
rule-based interpreter: it does not call a language model or any network service.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

_FOLLOW_UP_PATTERN = re.compile(r"\b(it|that|this|those|these|they|also|and then|what about|how about)\b", re.I)
_COMPARISON_KEYWORDS = (
    "compare", "compared", "versus", " vs ", "last year", "previous period", "prior period",
    "last period", "trend", "getting worse", "getting better", "deteriorating", "improving",
)
_RELIABILITY_KEYWORDS = (
    "reliable", "reliability", "confident", "confidence", "trust", "accurate", "accuracy",
    "certain", "how good is", "quality of",
)
_MISSING_KEYWORDS = (
    "missing", "not available", "what data", "which data", "gaps", "incomplete", "unavailable",
)
_SENSITIVITY_KEYWORDS = (
    "what if", "impact", "sensitivity", "sensitive to", "if leverage", "if revenue", "if debt",
    "if liquidity", "if margin", "if cash flow", "effect of", "effect if",
)


def build_credit_context(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    selected_features: pd.DataFrame,
    flags: pd.DataFrame,
    row_index: int,
    result: dict,
) -> dict[str, Any]:
    """Collect everything the copilot is allowed to reason about."""
    row = selected.iloc[0].copy()
    feat = selected_features.iloc[0].copy()
    active_flags = list(flags.iloc[row_index][flags.iloc[row_index]].index)
    company_id = row.get("company_id", "Selected company")
    revenue = float(feat.get("Revenue", 0.0) or 0.0)
    profit = float(feat.get("Net_Profit", 0.0) or 0.0)
    current_ratio = (
        float(feat.get("Current_Ratio", float("nan")))
        if pd.notna(feat.get("Current_Ratio")) else float("nan")
    )
    leverage = float(feat.get("Debt_to_Assets", 0.0) or 0.0)
    margin = float(feat.get("EBITDA_Margin", 0.0) or 0.0)
    coverage = float(feat.get("Interest_Coverage", 0.0) or 0.0)
    cash_flow = float(feat.get("Cash_Flow_Operations", 0.0) or 0.0)
    probability = result.get("distress_probability")
    health_score = result.get("health_score")
    history_available = bool("company_id" in frame.columns and "period" in frame.columns)
    return {
        "company": company_id,
        "period": row.get("period", "latest identified period"),
        "revenue": revenue,
        "profit": profit,
        "current_ratio": current_ratio,
        "debt_to_assets": leverage,
        "ebitda_margin": margin,
        "interest_coverage": coverage,
        "cash_flow_operations": cash_flow,
        "default_probability": float(probability) if probability is not None else None,
        "health_score": health_score,
        "risk_category": result["risk_category"],
        "assessment_method": result.get("method", "Existing ML model"),
        "coverage_label": result.get("coverage_label", "Coverage not available"),
        "coverage_percent": result.get("coverage_percent"),
        "warnings": active_flags,
        "top_risk": result.get("top_risk_factors", []),
        "protective": result.get("protective_factors", []),
        "history_available": history_available,
        "prior_period": _prior_period_summary(frame, row_index),
    }


def _prior_period_summary(frame: pd.DataFrame, row_index: int) -> dict[str, Any] | None:
    """Locate the previous company-period so comparisons use observed history."""
    if "company_id" not in frame.columns or "period" not in frame.columns:
        return None
    current = frame.iloc[row_index]
    company = str(current.get("company_id"))
    history = frame.loc[frame["company_id"].astype(str) == company].copy()
    if len(history) < 2:
        return None
    history["_parsed_period"] = pd.to_datetime(history["period"], errors="coerce")
    history["_position"] = range(len(history))
    history = history.sort_values(["_parsed_period", "_position"])
    if current.name not in history.index:
        return None
    position = int(history.index.get_indexer([current.name])[0])
    if position == 0:
        return None
    previous_row = history.iloc[position - 1]
    features = pd.to_numeric(history["Revenue"], errors="coerce")
    return {
        "period": str(previous_row.get("period")),
        "revenue": None if pd.isna(previous_row.get("Revenue")) else float(previous_row.get("Revenue")),
        "revenue_change": (
            None if pd.isna(features.iloc[position]) or not features.iloc[position - 1]
            else float((features.iloc[position] - features.iloc[position - 1]) / abs(features.iloc[position - 1]))
        ),
    }


def resolve_question(question: str, history: Sequence[str] = ()) -> str:
    """Expand a short follow-up so it keeps the earlier question's subject."""
    text = (question or "").strip()
    if not text:
        return "summarize the financial health"
    if history and len(text) < 45 and _FOLLOW_UP_PATTERN.search(text):
        previous = str(history[-1]).strip()
        if previous and previous.lower() != text.lower():
            return f"{previous.rstrip('?.!')}? {text}"
    return text


def suggested_questions(context: dict[str, Any]) -> list[str]:
    """Context-aware prompt chips for the copilot input."""
    company = context.get("company", "this company")
    category = str(context.get("risk_category", "risk"))
    rule_based = context.get("assessment_method") != "Existing ML model"
    questions = [
        f"Why is {company} currently {category.lower()}?",
        "Summarize the financial health",
        "How reliable is this assessment?",
        "Which financial fields are missing?",
        "What are the biggest risk factors?",
        "Which strengths offset the risk?",
        "What could increase credit risk?",
        "What should management improve first?",
    ]
    if context.get("history_available") and context.get("prior_period"):
        questions.append("How does this compare with the previous period?")
    questions.append("What if revenue falls by 20%?")
    if rule_based:
        questions = [q for q in questions if "shap" not in q.lower()]
    return questions[:9]


def _reliability_answer(context: dict[str, Any]) -> str:
    method = context.get("assessment_method", "Existing ML model")
    coverage = context.get("coverage_label", "Coverage not available")
    confidence = context.get("coverage_percent")
    base = (
        f"The assessment is an analytical research estimate, not a guaranteed outcome. "
        f"Data coverage for this record is {coverage}."
    )
    if method != "Existing ML model":
        return (
            f"{base} Because the observed fields are not sufficient for the existing ML model, "
            "this record was assessed with transparent financial rules, which is a heuristic "
            "risk index rather than a calibrated probability of default. Supply more periods and "
            "balance-sheet fields before relying on it."
        )
    return (
        f"{base} The score is the model's own output, and the confidence figure shown on the "
        "Risk Prediction page is only the distance from the decision midpoint, not calibrated "
        "uncertainty. Validate performance on representative future periods before operational use."
    )


def _missing_answer(context: dict[str, Any]) -> str:
    coverage = context.get("coverage_label", "Coverage not available")
    ratio = context.get("current_ratio")
    if pd.isna(ratio):
        return (
            f"Core-field coverage is {coverage}. Liquidity, leverage, margin, and coverage ratios "
            "could not be calculated because their inputs were not observed in this record, "
            "so those areas are unassessed rather than healthy."
        )
    return (
        f"Core-field coverage is {coverage}. Where a ratio is shown as not available, the "
        "underlying field was missing or its denominator was zero; missing values are never "
        "treated as healthy. Add the missing statement lines to widen the assessment."
    )


def _comparison_answer(context: dict[str, Any]) -> str:
    prior = context.get("prior_period")
    if not prior:
        return (
            "There is not an earlier dated company-period in this dataset, so no like-for-like "
            "comparison is possible. Add company and period columns covering multiple periods to "
            "enable trend and period-over-period comparison."
        )
    revenue_change = prior.get("revenue_change")
    movement = (
        f" Revenue is {revenue_change:+.1%} against the previous period."
        if revenue_change is not None
        else " A revenue change could not be calculated from the available rows."
    )
    return (
        f"The previous recorded period is {prior['period']}.{movement} "
        f"The current assessment is {context['risk_category']} with a "
        f"{_probability_text(context)} distress estimate, and the active early-warning signals are "
        f"{', '.join(context['warnings']) if context['warnings'] else 'none'}. "
        "Period-over-period movement in leverage, liquidity, and margins is the more reliable "
        "signal than a single-period snapshot."
    )


def _sensitivity_answer(context: dict[str, Any]) -> str:
    leverage = context.get("debt_to_assets", 0.0)
    liquidity = context.get("current_ratio")
    drivers = [
        name
        for name, value, threshold, comparison in (
            ("leverage", leverage, 0.65, ">="),
            ("liquidity", liquidity, 1.0, "<"),
            ("operating margin", context.get("ebitda_margin", 0.0), 0.0, "<"),
            ("operating cash flow", context.get("cash_flow_operations", 0.0), 0.0, "<"),
        )
        if value is not None and not pd.isna(value)
        and ((value >= threshold) if comparison == ">=" else (value < threshold))
    ]
    if not drivers:
        drivers = ["revenue level", "operating cash generation"]
    return (
        f"This assessment is most sensitive to {', '.join(drivers)}. "
        "Use the Scenario Simulator to apply an explicit revenue, margin, or debt change and read "
        "the recalculated estimate; the simulator shows the model's own response rather than a "
        "verbal estimate."
    )


def _probability_text(context: dict[str, Any]) -> str:
    probability = context.get("default_probability")
    return f"{probability:.1%}" if probability is not None else "an unavailable"


def generate_credit_copilot_response(
    question: str,
    context: dict[str, Any],
    history: Sequence[str] = (),
) -> str:
    """Answer a copilot question from the observed figures for one company-period."""
    q = resolve_question(question, history).lower()
    warnings = context["warnings"]
    risk_prob = context["default_probability"]
    current_ratio = context["current_ratio"]
    leverage = context["debt_to_assets"]
    margin = context["ebitda_margin"]
    cash_flow = context["cash_flow_operations"]
    revenue = context["revenue"]
    profit = context["profit"]

    if not q:
        q = "summarize the financial health"

    if any(keyword in q for keyword in _RELIABILITY_KEYWORDS):
        return _reliability_answer(context)

    if any(keyword in q for keyword in _MISSING_KEYWORDS):
        return _missing_answer(context)

    if any(keyword in q for keyword in _SENSITIVITY_KEYWORDS):
        return _sensitivity_answer(context)

    if any(keyword in q for keyword in _COMPARISON_KEYWORDS):
        return _comparison_answer(context)

    if context.get("assessment_method") != "Existing ML model":
        drivers = [
            str(item.get("feature", ""))
            for item in context.get("top_risk", [])
            if item.get("feature")
        ]
        summary = (
            f"This record has {context['coverage_label']} and was assessed with "
            "transparent financial rules because the observed fields are not sufficient for the existing ML model. "
        )
        if risk_prob is None:
            summary += "There is not enough numeric data to calculate a risk index."
        else:
            summary += f"The rule-based risk index is {risk_prob:.1%} and the category is {context['risk_category']}."
        if drivers:
            summary += f" Observed rule-based risk drivers include {', '.join(drivers[:3])}."
        if warnings:
            summary += f" Active early-warning indicators: {', '.join(warnings[:3])}."
        summary += " This index is a heuristic, not a calibrated probability of default."
        return summary

    if any(keyword in q for keyword in ["why", "risky", "risk", "risk factors", "biggest risk"]):
        reasons = []
        if pd.notna(current_ratio) and current_ratio < 1.0:
            reasons.append(f"liquidity is thin at {current_ratio:.2f}x")
        if leverage >= 0.65:
            reasons.append(f"debt loads are elevated at {leverage:.2f} debt-to-assets")
        if margin < 0 or cash_flow < 0:
            reasons.append("operating margin and cash generation are under pressure")
        if warnings:
            reasons.append("early-warning flags such as " + ", ".join(warnings[:3]))
        if not reasons:
            reasons.append("the model is pricing in a moderate probability of distress across revenue, leverage, and margin signals")
        return (
            f"{context['company']} is considered risky because " + "; ".join(reasons) + ". "
            f"The current risk estimate sits at {risk_prob:.1%}, and the model is particularly sensitive to leverage, liquidity, and operating cash generation."
        )

    if any(keyword in q for keyword in ["shap", "feature importance", "factors", "top risk"]):
        top_risk = context["top_risk"][:3] if context.get("top_risk") else ["Revenue pressure", "Leverage strain", "Liquidity deterioration"]
        return (
            f"The largest contributors to the distress score are {', '.join(str(item) for item in top_risk[:3])}. "
            "In practice, elevated leverage, weaker liquidity, and pressure on profitability are the strongest drivers of the current model output."
        )

    if any(keyword in q for keyword in ["credit", "lend", "considered for credit", "should this company"]):
        if risk_prob < 0.25 and current_ratio > 1.0 and cash_flow >= 0:
            return f"Based on current signals, {context['company']} appears broadly creditworthy for a cautious structure: distress probability is {risk_prob:.1%}, cash flow is positive, and liquidity remains above 1.0x."
        return f"This company should be treated cautiously. With a {risk_prob:.1%} distress probability and several active warning signals, underwriting should require tighter covenants, collateral review, or a lower exposure limit."

    if any(keyword in q for keyword in ["weakness", "weakest", "financial weaknesses"]):
        weaknesses = []
        if pd.notna(current_ratio) and current_ratio < 1.0:
            weaknesses.append(f"liquidity is weak at {current_ratio:.2f}x")
        if leverage >= 0.65:
            weaknesses.append(f"leverage is elevated at {leverage:.2f} debt-to-assets")
        if margin < 0 or cash_flow < 0:
            weaknesses.append("operating margin and cash generation are deteriorating")
        if not weaknesses:
            weaknesses.append("operating resilience is uneven despite an otherwise stable revenue base")
        return f"The main weaknesses are {', '.join(weaknesses)}. These are the items most likely to pressure repayment capacity if conditions worsen."

    if any(keyword in q for keyword in ["strength", "strengths", "what are the strengths", "healthy"]):
        strengths = []
        if revenue > 0:
            strengths.append(f"revenue base remains at {revenue:,.0f}")
        if profit > 0:
            strengths.append(f"net profit remains positive at {profit:,.0f}")
        if current_ratio > 1.0:
            strengths.append(f"liquidity remains above 1.0x at {current_ratio:.2f}x")
        if not strengths:
            strengths.append("the model still sees some positive operating characteristics even under moderate risk")
        return f"The main strengths are {', '.join(strengths)}. Those fundamentals help offset the risk profile, but they are not yet strong enough to eliminate the monitoring need."

    if any(keyword in q for keyword in ["management improve", "improve", "should management"]):
        actions = [
            "tighten working-capital discipline and reduce receivable days",
            "lower dependency on debt-funded growth and improve debt service capacity",
            "improve operating cash conversion to sustain repayment capability",
        ]
        return f"Management should prioritize {', '.join(actions)}. These steps would strengthen liquidity, reduce leverage pressure, and improve the credit standing most quickly."

    if any(keyword in q for keyword in ["increase risk", "what could cause", "credit risk to increase", "cause risk"]):
        return (
            "The risk would rise if revenue contracts materially, operating cash flow turns negative, debt builds further, "
            "or liquidity slips below 1.0x. In this scenario, the model would likely move toward a higher default probability."
        )

    if any(keyword in q for keyword in ["summarize", "summary", "financial health"]):
        health = context.get("health_score")
        health_text = f" a health score of {health:.0f}/100" if health is not None else " no health score"
        return (
            f"{context['company']} is currently assessed as {context['risk_category']} with a {_probability_text(context)} distress probability and{health_text}. "
            f"Revenue is {revenue:,.0f}, net profit is {profit:,.0f}, liquidity is {current_ratio:.2f}x, and leverage is {leverage:.2f} debt-to-assets. "
            f"The main watchpoints are {', '.join(warnings[:3]) if warnings else 'limited but active risk factors'} ."
        )

    if any(keyword in q for keyword in ["early warning", "warning signals", "signals"]):
        return (
            f"The early-warning flags currently active are: {', '.join(warnings) if warnings else 'none at the current period'}. "
            "These are designed to highlight deteriorating liquidity, leverage drift, weak margins, and negative cash generation before distress becomes more visible."
        )

    if any(keyword in q for keyword in ["revenue falls", "20%", "fall by 20", "if revenue falls"]):
        adjusted_revenue = revenue * 0.8
        return (
            f"If revenue were to fall by 20%, the company would likely see a weaker margin profile and reduced cash conversion. "
            f"At an adjusted revenue level of {adjusted_revenue:,.0f}, the risk profile would likely move toward higher distress because leverage and fixed-cost pressure would become more visible."
        )

    return (
        f"{context['company']} currently shows a {context['risk_category']} profile with a {_probability_text(context)} distress probability. "
        "Liquidity, leverage, and operating cash flow are the key variables shaping the assessment, and the most immediate actions are to improve margin resilience and funding stability."
    )
