import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from msme_ews.features import MODEL_FEATURES, engineer_features
from msme_ews.prediction import predict_financial_health, risk_category


def test_prediction_response_and_risk_categories():
    row = pd.DataFrame([{
        "Revenue": 100000, "EBITDA": 8000, "Net_Profit": 2000, "Total_Assets": 70000,
        "Total_Liabilities": 40000, "Current_Assets": 30000, "Current_Liabilities": 18000,
        "Cash_Flow_Operations": 5000, "Debt": 25000, "Interest_Expense": 2000,
        "Accounts_Receivable": 9000, "Inventory": 7000,
    }])
    model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("classifier", DummyClassifier(strategy="prior"))])
    model.fit(pd.concat([engineer_features(row)] * 4, ignore_index=True), [0, 0, 1, 1])
    response = predict_financial_health(row, bundle={"model": model, "features": list(MODEL_FEATURES)}, include_explanations=False)
    assert {"distress_probability", "risk_category", "top_risk_factors", "protective_factors"}.issubset(response)
    assert 0 <= response["distress_probability"] <= 1
    assert risk_category(0.1) == "Low Risk"
    assert risk_category(0.4) == "Moderate Risk"
    assert risk_category(0.8) == "High Risk"


def test_prediction_rejects_multirow_input():
    try:
        predict_financial_health(pd.DataFrame({"Revenue": [10, 20]}), bundle={}, include_explanations=False)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("Multirow prediction should be rejected")