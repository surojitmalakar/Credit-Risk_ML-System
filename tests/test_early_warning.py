import pandas as pd

from msme_ews.early_warning import early_warning_indicators


def test_historical_deterioration_triggers_trend_warnings():
    frame = pd.DataFrame({
        "company_id": ["A", "A"],
        "period": ["2024-12-31", "2025-12-31"],
        "Revenue": [1000, 800],
        "EBITDA": [100, 50],
        "Net_Profit": [50, 20],
        "Total_Assets": [500, 500],
        "Total_Liabilities": [200, 200],
        "Current_Assets": [200, 100],
        "Current_Liabilities": [100, 100],
        "Cash_Flow_Operations": [20, -2],
        "Debt": [150, 250],
        "Interest_Expense": [10, 10],
        "Accounts_Receivable": [100, 240],
        "Inventory": [50, 50],
    })
    flags = early_warning_indicators(frame)
    last = flags.iloc[1]

    assert last["Rapid revenue decline"]
    assert last["Increasing leverage"]
    assert last["Falling liquidity"]
    assert last["Deteriorating margins"]
    assert last["Negative operating cash flow"]
    assert last["Increasing receivable days"]
    assert last["Falling interest coverage"]