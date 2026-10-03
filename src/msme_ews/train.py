"""Command-line entry point for reproducible model training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from msme_ews.data import validate_financial_data
from msme_ews.modeling import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--target", default="distress_label", help="Binary target column")
    parser.add_argument("--protected-attribute", help="Optional audit-only group column")
    parser.add_argument("--output", default="models/msme_model.joblib", help="Model bundle path")
    args = parser.parse_args()
    data = validate_financial_data(pd.read_csv(args.data))
    bundle = train_models(data, args.target, args.protected_attribute)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"Saved {bundle['model_name']} model bundle to {output}")
    print(f"Evaluation note: {bundle['report']['evaluation_note']}")
    print("Held-out metrics:")
    for metric, score in bundle["report"]["test_metrics"].items():
        print(f"  {metric}: {score}")
    print("Cross-validation means:")
    for model_name, metrics in bundle["report"]["cv_metrics"].items():
        formatted = ", ".join(f"{metric}={score:.3f}" for metric, score in metrics.items())
        print(f"  {model_name}: {formatted}")


if __name__ == "__main__":
    main()