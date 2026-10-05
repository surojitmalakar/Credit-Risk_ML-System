"""Deterministic synthetic data for exercising the project; not empirical data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(companies: int = 240, periods: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create synthetic company-periods with deliberately illustrative labels."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for company_number in range(companies):
        base_revenue = rng.lognormal(mean=11.2, sigma=0.75)
        base_margin = rng.normal(0.13, 0.09)
        leverage = rng.uniform(0.12, 0.78)
        owner_gender = rng.choice(["woman", "man"], p=[0.32, 0.68])
        sector = rng.choice(["manufacturing", "retail", "services", "agriculture"])
        company_id = f"SYN-{company_number:04d}"
        prior_revenue = None
        for year in range(periods):
            growth = rng.normal(0.025, 0.13) - (0.035 if year else 0)
            revenue = base_revenue * (1 + growth) ** year * np.exp(rng.normal(0, 0.08))
            margin = base_margin + rng.normal(0, 0.035) - year * rng.uniform(0, 0.012)
            assets = revenue * rng.uniform(0.42, 1.2)
            liabilities = assets * leverage
            debt = liabilities * rng.uniform(0.35, 0.9)
            current_assets = assets * rng.uniform(0.18, 0.58)
            current_liabilities = liabilities * rng.uniform(0.3, 0.8)
            ebitda = revenue * margin
            net_profit = ebitda - revenue * rng.uniform(0.025, 0.095)
            cash_flow = net_profit + rng.normal(0, revenue * 0.045)
            interest = max(debt * rng.uniform(0.035, 0.12), 1)
            receivables = revenue * rng.uniform(0.035, 0.22)
            inventory = revenue * rng.uniform(0.02, 0.24)
            actual_growth = revenue / prior_revenue - 1 if prior_revenue else growth
            prior_revenue = revenue
            stress = (
                1.45 * max(0, debt / max(assets - liabilities, assets * 0.03) - 1.1)
                + 1.5 * max(0, 0.85 - current_assets / max(current_liabilities, 1))
                + 1.1 * max(0, -margin)
                + max(0, -cash_flow / max(revenue, 1))
                + 0.8 * max(0, -actual_growth)
            )
            probability = 1 / (1 + np.exp(-(stress - 0.75) * 1.7))
            rows.append({
                "company_id": company_id, "period": f"{2021 + year}-12-31", "sector": sector,
                "owner_gender": owner_gender, "Revenue": round(revenue, 2),
                "EBITDA": round(ebitda, 2), "Net_Profit": round(net_profit, 2),
                "Total_Assets": round(assets, 2), "Total_Liabilities": round(liabilities, 2),
                "Current_Assets": round(current_assets, 2), "Current_Liabilities": round(current_liabilities, 2),
                "Cash_Flow_Operations": round(cash_flow, 2), "Debt": round(debt, 2),
                "Interest_Expense": round(interest, 2), "Accounts_Receivable": round(receivables, 2),
                "Inventory": round(inventory, 2), "Sales_Growth": round(actual_growth, 4),
                "distress_label": int(rng.random() < probability),
            })
    frame = pd.DataFrame(rows)
    frame["dataset_source"] = "synthetic_demo"
    frame.attrs["dataset_kind"] = "synthetic_demo"
    return frame


def write_demo_data(path: str | Path, companies: int = 240, periods: int = 3) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_demo_data(companies=companies, periods=periods).to_csv(output, index=False)
    return output


def main() -> None:
    """Write a synthetic CSV for development and tests; the app never loads it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="data/demo_msme.csv", help="Destination CSV path")
    parser.add_argument("--companies", type=int, default=240)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = make_demo_data(companies=args.companies, periods=args.periods, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} synthetic company-periods to {output}")
    print(f"Distress label rate: {frame['distress_label'].mean():.1%}")
    print("Synthetic data only; do not use for real credit decisions.")


if __name__ == "__main__":
    main()