from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import load_runtime_settings
from .predict import predict_records


class Application(BaseModel):
    checking_status: str
    duration_months: int = Field(gt=0)
    credit_history: str
    purpose: str
    credit_amount: float = Field(gt=0)
    savings_status: str
    employment_since: str
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: str
    other_debtors: str
    residence_since: int = Field(ge=1, le=4)
    property: str
    age: int = Field(ge=18, le=100)
    other_installment_plans: str
    housing: str
    existing_credits: int = Field(ge=1)
    job: str
    dependents: int = Field(ge=1)
    telephone: str
    foreign_worker: str


app = FastAPI(title="Credit Risk ML API", version="0.1.0")


@app.get("/app", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "ui" / "index.html")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "credit-risk-ml-api",
        "status": "ok",
        "app": "/app",
        "docs": "/docs",
        "health": "/health",
        "prediction": "POST /predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(application: Application) -> dict[str, Any]:
    try:
        return predict_records([application.model_dump()], load_runtime_settings().artifact_dir)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No production model is registered") from exc
