"""Portfolio-intelligence summary over screened records.

Uses the existing vectorized screening scores; never fabricates entities.
Single-entity datasets return ``multi_entity=False`` with an explanation.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _key(value: object) -> str:
    return re.sub(r"[\s_\-]+", " ", str(value).strip().lower())


def _entity_key(frame: pd.DataFrame) -> pd.Series:
    for col in ("company_id", "customer_id", "customer_name", "company", "borrower_id", "loan_id"):
        if col in frame.columns:
            return frame[col].astype("string").fillna("Unknown")
    return pd.Series(["All records"] * len(frame), index=frame.index)


def portfolio_intelligence(frame: pd.DataFrame, scores: pd.DataFrame) -> dict[str, Any]:
    entities = _entity_key(frame)
    n_entities = int(entities.nunique())
    out: dict[str, Any] = {"entities": n_entities, "multi_entity": n_entities > 1}
    if scores is None or scores.empty:
        out.update({"available": False, "reason": "No screening scores available."})
        return out
    out["available"] = True
    cats = scores["Risk category"].astype("string") if "Risk category" in scores else pd.Series(dtype="string")
    idx = pd.to_numeric(scores.get("Risk index (0-100)"), errors="coerce") if "Risk index (0-100)" in scores else pd.Series(dtype=float)
    health = pd.to_numeric(scores.get("Health score"), errors="coerce") if "Health score" in scores else pd.Series(dtype=float)
    out["risk_counts"] = {str(k): int((cats == k).sum()) for k in cats.dropna().unique()}
    out["avg_health"] = float(health.mean()) if health.notna().any() else None
    out["avg_risk"] = float(idx.mean()) if idx.notna().any() else None
    per_entity = pd.DataFrame({"entity": entities.to_numpy(),
                               "risk": idx.to_numpy() if len(idx) == len(frame) else [float("nan")] * len(frame),
                               "band": cats.to_numpy() if len(cats) == len(frame) else [""] * len(frame)})
    grp = per_entity.groupby("entity", dropna=False)["risk"].mean().sort_values(ascending=False)
    out["riskiest"] = [(str(k), float(v)) for k, v in grp.head(10).items() if pd.notna(v)]
    out["strongest"] = [(str(k), float(v)) for k, v in grp.tail(10).iloc[::-1].items() if pd.notna(v)]
    # Concentration: share of records in top band / top entity.
    top_share = float((cats == "High Risk").mean()) if len(cats) else None
    out["high_share"] = top_share
    ent_counts = entities.value_counts(normalize=True)
    out["largest_entity_share"] = float(ent_counts.iloc[0]) if len(ent_counts) else None
    out["largest_entity"] = str(ent_counts.index[0]) if len(ent_counts) else None
    # Segment / geographic concentration from low-cardinality columns.
    conc = []
    for col in frame.columns:
        if col in {"company_id", "customer_id", "company"}:
            continue
        nun = frame[col].nunique(dropna=True)
        if 2 <= nun <= 12 and len(frame) >= 5:
            top = frame[col].astype("string").value_counts(normalize=True).head(5)
            conc.append({"column": str(col),
                         "top": "; ".join(f"{i} {v:.0%}" for i, v in top.items())})
    out["concentrations"] = conc[:6]
    # Anomaly concentration: flagged share per band when available.
    out["bands"] = {str(k): int(v) for k, v in cats.value_counts(dropna=False).items()}
    # Explicit low / moderate / high / critical tier counts.
    tiers = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0, "Critical Risk": 0}
    for category, count in out["bands"].items():
        if category in tiers:
            tiers[category] = int(count)
    out["risk_tiers"] = tiers
    # Geographic concentration from recognised location columns.
    geo_columns = [
        column for column in frame.columns
        if _key(column) in {"country", "state", "city", "region",
                            "geography", "nation", "province", "district"}
    ]
    geo = []
    for column in geo_columns:
        top = frame[column].astype("string").value_counts(normalize=True).head(5)
        geo.append({"column": str(column),
                    "top": "; ".join(f"{i} {v:.0%}" for i, v in top.items())})
    out["geo_concentration"] = geo
    # Anomaly concentration when an anomaly score/flag column is present.
    anomaly_column = next(
        (column for column in frame.columns
         if "anomal" in _key(column) or "outlier" in _key(column)),
        None,
    )
    if anomaly_column is not None:
        flag = pd.to_numeric(frame[anomaly_column], errors="coerce")
        if flag.notna().any():
            anomalous = float((flag.fillna(0) != 0).mean())
            out["anomaly_concentration"] = {"column": str(anomaly_column), "share": anomalous}
        else:
            out["anomaly_concentration"] = None
    else:
        out["anomaly_concentration"] = None
    # Portfolio trend: mean risk index by period when a period column exists.
    period_column = next(
        (column for column in frame.columns
         if _key(column) in {"period", "year", "date", "quarter", "month"}),
        None,
    )
    if period_column is not None and len(idx) == len(frame):
        trend_frame = pd.DataFrame({
            "period": frame[period_column].astype("string").to_numpy(),
            "risk": idx.to_numpy(),
        })
        trend = (
            trend_frame.groupby("period", dropna=False)["risk"]
            .agg(["mean", "count"])
            .reset_index()
        )
        out["trend"] = [
            {"period": str(row["period"]),
             "mean_risk": (float(row["mean"]) if pd.notna(row["mean"]) else None),
             "records": int(row["count"])}
            for _, row in trend.iterrows()
        ]
    else:
        out["trend"] = None
    return out
