import numpy as np
import pandas as pd

from msme_ews.data import validate_financial_data
from msme_ews.features import engineer_features


def sample_frame():
    return pd.DataFrame({
        "Revenue": [1000.0, 0.0], "EBITDA": [100.0, 20.0], "Net_Profit": [50.0, -10.0],
        "Total_Assets": [500.0, 400.0], "Total_Liabilities": [250.0, 450.0],
        "Current_Assets": [200.0, 100.0], "Current_Liabilities": [100.0, 0.0],
        "Cash_Flow_Operations": [25.0, -5.0], "Debt": [150.0, 300.0],
        "Interest_Expense": [10.0, 0.0], "Accounts_Receivable": [100.0, 10.0],
        "Inventory": [50.0, 20.0],
    })


def test_validation_normalizes_common_column_aliases():
    frame = validate_financial_data(pd.DataFrame({"Annual Revenue": [123], "Total Assets": [99]}))
    assert list(frame.columns) == ["Revenue", "Total_Assets"]


def test_validation_rejects_unrecognized_or_empty_data():
    for frame in (pd.DataFrame(), pd.DataFrame({"notes": ["not financial"]})):
        try:
            validate_financial_data(frame)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input should be rejected")


def test_ratios_calculate_and_zero_denominators_become_missing():
    features = engineer_features(sample_frame())
    assert features.loc[0, "Current_Ratio"] == 2.0
    assert features.loc[0, "Quick_Ratio"] == 1.5
    assert features.loc[0, "Debt_to_Equity"] == 0.6
    assert features.loc[0, "Interest_Coverage"] == 10.0
    assert np.isnan(features.loc[1, "Current_Ratio"])
    assert features.loc[1, "Debt_to_Equity"] == -6.0


def test_abnormal_ratio_values_are_bounded_and_schema_is_stable():
    frame = sample_frame()
    frame.loc[0, "Current_Liabilities"] = 0.001
    features = engineer_features(frame)
    assert features.loc[0, "Current_Ratio"] <= 100
    assert features.columns.is_unique