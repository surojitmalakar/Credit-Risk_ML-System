"""Optional FastAPI wrapper for one-row financial distress predictions."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from msme_ews.prediction import DEFAULT_MODEL_PATH, load_model_bundle, predict_financial_health


class PredictionRequest(BaseModel):
    financial_data: dict[str, Any]


def create_app() -> FastAPI:
    application = FastAPI(title="MSME Financial Distress Early-Warning API",
                          description="Research estimates only; not a lending decision service.", version="0.1.0")

    @application.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        try:
            bundle = load_model_bundle(DEFAULT_MODEL_PATH)
            if bundle.get("is_demo") is True:
                raise HTTPException(
                    status_code=503,
                    detail="The configured model is synthetic demo data; train a model on reviewed labeled data before using the API.",
                )
            return predict_financial_health(request.financial_data, bundle=bundle)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail="Train a model before using the API.") from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application


app = create_app()