from msme_ews.demo import make_demo_data
from msme_ews.modeling import train_models


def test_grouped_training_compares_models_without_protected_features():
    data = make_demo_data(companies=100, periods=2)
    bundle = train_models(data, protected_attribute="owner_gender")
    report = bundle["report"]

    assert set(report["cv_metrics"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert set(report["test_metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert report["test_metrics"]["roc_auc"] is not None
    assert report["test_companies"] < data["company_id"].nunique()
    assert "owner_gender" not in bundle["features"]
    assert set(report["fairness"]) == {"woman", "man"}
    assert "illustrative" in report["evaluation_note"]