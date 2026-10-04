"""Multi-factor stress testing for one company-period.

Scenarios are built as a frame of adjusted copies of a single record so the whole
grid is scored in one model call instead of one call per combination. Every
result stays an analytical estimate from the existing model or, for records with
too few observed fields, the transparent rule-based index.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from msme_ews.financial_analysis import rule_based_risk_index
from msme_ews.features import engineer_features
from msme_ews.prediction import risk_category

# Cutoffs mirror risk_category so a grid and the single-record view agree.
DEFAULT_BANDS = (0.20, 0.40, 0.70)


@dataclass(frozen=True)
class StressFactor:
    """One lever the user can shock, with the levels offered in the UI.

    ``steps`` are displayed values (percent or percentage points). ``shock``
    converts a displayed level into the fraction that :func:`apply_stress`
    expects, which keeps the stress engine consistent with the existing
    single-scenario simulator.
    """

    key: str
    label: str
    unit: str
    steps: tuple[float, ...]
    percent: bool = True

    @property
    def kind(self) -> str:
        return "percent" if self.percent else "points"

    def shock(self, step: float) -> float:
        return float(step) / 100 if self.percent else float(step)

    def format_step(self, step: float) -> str:
        return f"{step:+g}%" if self.percent else f"{step:+g} pp"


STRESS_FACTORS: tuple[StressFactor, ...] = (
    StressFactor("revenue_change", "Revenue", "%", (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0)),
    StressFactor("margin_change", "Operating margin", "pp", (-8.0, -4.0, 0.0, 4.0, 8.0), percent=False),
    StressFactor("debt_change", "Debt", "%", (-20.0, -10.0, 0.0, 10.0, 20.0)),
    StressFactor("interest_change", "Interest cost", "%", (-50.0, 0.0, 50.0, 100.0)),
    StressFactor("current_liability_change", "Current liabilities", "%", (-20.0, 0.0, 20.0, 40.0)),
    StressFactor("receivable_change", "Receivables", "%", (-30.0, -10.0, 0.0, 10.0)),
)

# Preset shocks are expressed as fractions of the modelled fields, matching
# apply_scenario_adjustments, so they are not converted.
STRESS_PRESETS: dict[str, dict[str, float]] = {
    "Mild downturn": {"revenue_change": -0.10, "margin_change": -0.02},
    "Revenue shock": {"revenue_change": -0.20, "margin_change": -0.04, "debt_change": 0.10},
    "Rate and liquidity shock": {
        "revenue_change": -0.05,
        "margin_change": -0.02,
        "debt_change": 0.05,
        "interest_change": 0.50,
        "current_liability_change": 0.20,
    },
    "Severe combined": {
        "revenue_change": -0.30,
        "margin_change": -0.08,
        "debt_change": 0.20,
        "interest_change": 1.00,
        "current_liability_change": 0.40,
        "receivable_change": 0.30,
    },
}

# The columns each lever needs in the source record.
_REQUIRED_COLUMNS = {
    "revenue_change": ("Revenue", "EBITDA"),
    "margin_change": ("Revenue", "EBITDA"),
    "debt_change": ("Debt",),
    "interest_change": ("Interest_Expense",),
    "current_liability_change": ("Current_Liabilities", "Current_Assets"),
    "receivable_change": ("Accounts_Receivable",),
}


def available_factors(row: pd.Series | pd.DataFrame) -> list[StressFactor]:
    """Levers whose source fields are actually observed for this record."""
    observed = set(row.columns) if isinstance(row, pd.DataFrame) else set(row.index)
    usable: list[StressFactor] = []
    for factor in STRESS_FACTORS:
        required = _REQUIRED_COLUMNS[factor.key]
        if all(column in observed for column in required) and all(
            pd.notna(row[column]) if not isinstance(row, pd.DataFrame) else pd.notna(row[column].iloc[0])
            for column in required
        ):
            usable.append(factor)
    return usable


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def apply_stress(frame: pd.DataFrame, changes: dict[str, Any]) -> pd.DataFrame:
    """Apply one set of proportional shocks to every row of ``frame``.

    Each shock may be a scalar or a per-row sequence, so a full scenario grid is
    applied in a single vectorized pass. Units are fractions of the shocked field
    (0.20 means +20%), matching
    :func:`msme_ews.credit_assessment.apply_scenario_adjustments`.
    """
    adjusted = frame.copy()
    index = adjusted.index

    def level(key: str, default: float = 0.0) -> pd.Series:
        value = changes.get(key, default)
        if np.isscalar(value):
            return pd.Series(float(value), index=index, dtype="float64")
        return pd.Series(np.asarray(value, dtype="float64"), index=index)

    revenue = _numeric(adjusted, "Revenue")
    ebitda = _numeric(adjusted, "EBITDA")
    revenue_change = level("revenue_change")
    margin_change = level("margin_change")

    base_margin = np.where(revenue.to_numpy() != 0, ebitda.to_numpy() / revenue.to_numpy(), np.nan)
    adjusted_revenue = revenue * (1 + revenue_change)
    adjusted.loc[:, "Revenue"] = adjusted_revenue
    adjusted.loc[:, "EBITDA"] = adjusted_revenue * (base_margin + margin_change.to_numpy())

    if "Debt" in adjusted.columns:
        adjusted.loc[:, "Debt"] = _numeric(adjusted, "Debt") * (1 + level("debt_change"))
    if "Interest_Expense" in adjusted.columns:
        adjusted.loc[:, "Interest_Expense"] = _numeric(adjusted, "Interest_Expense") * (1 + level("interest_change"))
    if "Current_Liabilities" in adjusted.columns:
        adjusted.loc[:, "Current_Liabilities"] = _numeric(adjusted, "Current_Liabilities") * (1 + level("current_liability_change"))
    if "Accounts_Receivable" in adjusted.columns:
        adjusted.loc[:, "Accounts_Receivable"] = _numeric(adjusted, "Accounts_Receivable") * (1 + level("receivable_change"))

    if "Sales_Growth" in adjusted.columns:
        growth = pd.to_numeric(adjusted["Sales_Growth"], errors="coerce")
        adjusted.loc[:, "Sales_Growth"] = (1 + growth) * (1 + revenue_change) - 1
    if "Equity_Value" in adjusted.columns:
        adjusted.loc[:, "Equity_Value"] = _numeric(adjusted, "Total_Assets") - _numeric(adjusted, "Total_Liabilities")
    return adjusted


def scenario_label(changes: dict[str, float]) -> str:
    parts = [
        f"{factor.label} {factor.format_step(changes[factor.key])}"
        for factor in STRESS_FACTORS
        if factor.key in changes and changes[factor.key]
    ]
    return " · ".join(parts) if parts else "No change (current reported values)"


# How many scenario combinations one grid may evaluate before the app asks the
# user to narrow the selected factors.
MAX_GRID_SCENARIOS = 1_500


def build_grid(
    base_row: pd.DataFrame,
    factors: Sequence[StressFactor],
    steps: dict[str, Iterable[float]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Stack one adjusted copy of ``base_row`` per combination of factor levels.

    The base record is replicated once and every combination is applied in a
    single vectorized pass, which keeps a large grid fast. Returned specs use the
    displayed units so scenario labels match what the user selected.
    """
    levels = [
        list(steps.get(factor.key, factor.steps)) if steps else list(factor.steps)
        for factor in factors
    ]
    combinations = list(product(*levels)) if factors else [()]
    specs = [
        {factor.key: float(step) for factor, step in zip(factors, combination, strict=True)}
        for combination in combinations
    ]
    stacked = pd.concat([base_row] * len(specs), ignore_index=True)
    shocks = {
        factor.key: np.asarray([factor.shock(spec.get(factor.key, 0.0)) for spec in specs], dtype="float64")
        for factor in factors
    }
    return apply_stress(stacked, shocks), specs


def grid_size(factors: Sequence[StressFactor], steps: dict[str, Iterable[float]] | None = None) -> int:
    """Number of combinations a factor selection would evaluate."""
    total = 1
    for factor in factors:
        total *= len(list(steps.get(factor.key, factor.steps)) if steps else factor.steps)
    return total


def score_grid(
    grid: pd.DataFrame,
    bundle: dict[str, Any] | None,
    scorer: Callable[[pd.DataFrame], float | None] | None = None,
) -> pd.Series:
    """Return a transparent rule-based risk index in points, never a probability."""
    del bundle, scorer
    return rule_based_risk_index(grid, engineer_features(grid))["Rule risk index"].astype("float64")


def band_of(risk_index: float | None, bands: Sequence[float] = DEFAULT_BANDS) -> str:
    """Band a 0-100 rule index using cutoffs expressed as fractions."""
    if risk_index is None or pd.isna(risk_index):
        return "Insufficient Data"
    moderate, high, critical = (cutoff * 100 for cutoff in bands)
    if risk_index >= critical:
        return "Critical Risk"
    if risk_index >= high:
        return "High Risk"
    if risk_index >= moderate:
        return "Moderate Risk"
    return "Low Risk"


def run_stress_grid(
    base_row: pd.DataFrame,
    factors: Sequence[StressFactor],
    bundle: dict[str, Any] | None = None,
    bands: Sequence[float] = DEFAULT_BANDS,
    steps: dict[str, Iterable[float]] | None = None,
) -> pd.DataFrame:
    """Score a full multi-factor grid in one pass."""
    base_risk_index = float(score_grid(base_row, bundle).iloc[0])
    grid, specs = build_grid(base_row, factors, steps)
    risk_indices = score_grid(grid, bundle)
    results = pd.DataFrame({
        "Scenario": [scenario_label(spec) for spec in specs],
        "Risk index (0-100)": risk_indices.round(1),
        "Risk category": [band_of(value, bands) for value in risk_indices],
        "Change vs baseline (points)": (risk_indices - base_risk_index).round(1),
    })
    for factor in factors:
        results[factor.label] = [
            spec.get(factor.key, 0.0) for spec in specs
        ]
    results.attrs["base_risk_index"] = base_risk_index
    results.attrs["base_category"] = band_of(base_risk_index, bands)
    return results


def tornado_rows(
    base_row: pd.DataFrame,
    factors: Sequence[StressFactor],
    bundle: dict[str, Any] | None = None,
    bands: Sequence[float] = DEFAULT_BANDS,
) -> pd.DataFrame:
    """Single-factor sensitivity of the risk estimate, sorted by total impact."""
    base_risk_index = float(score_grid(base_row, bundle).iloc[0])
    levels = [(factor, factor.steps[0], factor.steps[-1]) for factor in factors]
    if not levels:
        return pd.DataFrame()
    # Each pair of rows isolates one factor; every other row keeps the reported values.
    shocks: dict[str, np.ndarray] = {}
    for position, (factor, downside, upside) in enumerate(levels):
        column = np.zeros(2 * len(levels), dtype="float64")
        column[2 * position] = factor.shock(downside)
        column[2 * position + 1] = factor.shock(upside)
        shocks[factor.key] = column
    probe = apply_stress(pd.concat([base_row] * (2 * len(levels)), ignore_index=True), shocks)
    risk_indices = score_grid(probe, bundle).to_numpy()
    rows = []
    for position, (factor, downside, upside) in enumerate(levels):
        down_value = float(risk_indices[2 * position])
        up_value = float(risk_indices[2 * position + 1])
        rows.append({
            "Factor": factor.label,
            f"Downside {factor.format_step(downside)}": down_value,
            f"Upside {factor.format_step(upside)}": up_value,
            "Downside impact": down_value - base_risk_index,
            "Upside impact": up_value - base_risk_index,
            "Worst band": band_of(max(down_value, up_value), bands),
            "Best band": band_of(min(down_value, up_value), bands),
        })
    frame = pd.DataFrame(rows)
    frame["Total impact"] = frame["Downside impact"].abs() + frame["Upside impact"].abs()
    return frame.sort_values("Total impact", ascending=False).reset_index(drop=True)


def summarize_stress(results: pd.DataFrame, base_risk_index: float) -> dict[str, Any]:
    """Headline figures for the stress-test header."""
    if results.empty:
        return {"scenarios": 0, "worst_risk_index": None, "best_risk_index": None, "elevated": 0}
    risk_index = results["Risk index (0-100)"]
    worst_index = risk_index.idxmax()
    best_index = risk_index.idxmin()
    return {
        "scenarios": int(len(results)),
        "worst_risk_index": float(risk_index.max()),
        "best_risk_index": float(risk_index.min()),
        "worst_scenario": str(results.loc[worst_index, "Scenario"]),
        "best_scenario": str(results.loc[best_index, "Scenario"]),
        "worst_band": band_of(float(risk_index.max())),
        "mean_change": float((risk_index - base_risk_index).mean()),
        "elevated": int((risk_index >= 60).sum()),
        "worse_than_reported": int((risk_index > base_risk_index + 1e-9).sum()),
    }


def breakeven_step(
    base_row: pd.DataFrame,
    factor: StressFactor,
    bundle: dict[str, Any] | None = None,
    target: float = 0.60,
    direction: str = "downside",
) -> float | None:
    """Largest single-lever move that still keeps the estimate below ``target``.

    Returns the displayed level, or ``None`` when no offered level stays under
    the target. A tree-based model response is not monotonic in the shocked
    inputs, so the offered levels are searched directly rather than solved for.
    """
    candidates = [
        step for step in factor.steps
        if (step < 0) == (direction == "downside") and step != 0
    ]
    candidates.sort(reverse=direction == "downside")
    if not candidates:
        return None
    shocks = [factor.shock(step) for step in candidates]
    probe = apply_stress(
        pd.concat([base_row] * len(candidates), ignore_index=True),
        {factor.key: np.asarray(shocks, dtype="float64")},
    )
    risk_indices = score_grid(probe, bundle).to_numpy()
    safe = [
        step for step, risk_index in zip(candidates, risk_indices, strict=True)
        if risk_index < target * 100
    ]
    return max(safe, key=abs) if safe else None


def category(risk_index: float | None) -> str:
    """Re-export the band helper used by the app for readable summaries."""
    return band_of(risk_index) if risk_index is not None else "Insufficient Data"