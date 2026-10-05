"""Enriched early-warning details derived from observed values only."""

from __future__ import annotations

import numpy as np
import pandas as pd

_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
_META = {
    "Rapid revenue decline": ("Sales growth <= -15%", "Sharp revenue contraction weakens cash generation.", "Verify periods and investigate the operating change."),
    "Negative operating cash flow": ("Operating cash flow < 0", "Core business consumed cash this period.", "Review cash conversion and working capital."),
    "Increasing leverage": ("Debt/assets >= 0.65 or +5pp vs prior", "High leverage limits shock absorption.", "Verify debt scope and maturities."),
    "Falling liquidity": ("Current ratio < 1.0x or -10% vs prior", "Obligations exceed current assets.", "Review near-term obligations and assets."),
    "Deteriorating margins": ("EBITDA/net margin < 0 or falling", "Cost pressure or pricing weakness.", "Confirm costs and margin drivers."),
    "Increasing receivable days": ("Receivable days > 90 or +10 vs prior", "Slow collection ties up cash.", "Review overdue receivables."),
    "Falling interest coverage": ("Interest coverage < 1.5x or falling", "Earnings barely cover interest.", "Verify interest and debt service."),
}
_FEAT = {
    "Rapid revenue decline": ("Sales_Growth", "{v:.1%}", "Sales growth"),
    "Negative operating cash flow": ("Cash_Flow_Operations", "{v:,.0f}", "Operating cash flow"),
    "Increasing leverage": ("Debt_to_Assets", "{v:.2f}x", "Debt/assets"),
    "Falling liquidity": ("Current_Ratio", "{v:.2f}x", "Current ratio"),
    "Deteriorating margins": ("EBITDA_Margin", "{v:.1%}", "EBITDA margin"),
    "Increasing receivable days": ("Receivable_Days", "{v:.1f}d", "Receivable days"),
    "Falling interest coverage": ("Interest_Coverage", "{v:.2f}x", "Interest coverage"),
}
_SEV = {
    "Rapid revenue decline": "CRITICAL", "Negative operating cash flow": "CRITICAL",
    "Increasing leverage": "HIGH", "Falling liquidity": "HIGH",
    "Falling interest coverage": "HIGH", "Deteriorating margins": "MEDIUM",
    "Increasing receivable days": "MEDIUM",
}


def detailed_warnings(frame, features, flags, row_index) -> list[dict]:
    if len(frame) == 0 or row_index < 0 or row_index >= len(frame):
        return []
    active = set()
    if flags is not None and len(flags) == len(frame):
        rf = flags.iloc[row_index]
        active = set(rf[rf].index.tolist())
    feat = features.iloc[row_index] if len(features) > row_index else pd.Series(dtype=float)
    raw = frame.iloc[row_index]
    rows = []
    for sig, (thr, why, act) in _META.items():
        fname, tmpl, label = _FEAT[sig]
        obs = None
        for src in (feat, raw):
            if fname in getattr(src, "index", []):
                try:
                    c = float(pd.to_numeric(pd.Series([src.get(fname)]), errors="coerce").iloc[0])
                except (TypeError, ValueError):
                    c = None
                if c is not None and np.isfinite(c):
                    obs = c
                    break
        fired = sig in active
        if fired and obs is None:
            obs = float("nan")
        if obs is None or (isinstance(obs, float) and np.isnan(obs)):
            rows.append({"Indicator": sig, "Actual value": "Not available",
                         "Threshold/reference": thr, "Severity": "INFO",
                         "Status": "Not assessed", "Why it matters": why,
                         "Recommended action": f"Provide {label} to assess."})
            continue
        try:
            aval = tmpl.format(v=float(obs)) if np.isfinite(obs) else "Not available"
        except (ValueError, TypeError):
            aval = "Not available"
        sev = _SEV.get(sig, "MEDIUM") if fired else "LOW"
        rows.append({"Indicator": sig, "Actual value": f"{label}: {aval}",
                     "Threshold/reference": thr, "Severity": sev,
                     "Status": "Triggered" if fired else "Not triggered",
                     "Why it matters": why, "Recommended action": act})
    rows.sort(key=lambda r: (_ORDER.get(r["Severity"], 9), r["Indicator"]))
    return rows


def triggered_warnings(detailed: list[dict]) -> list[dict]:
    return [r for r in detailed if r.get("Status") == "Triggered"]
