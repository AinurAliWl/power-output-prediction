"""FastAPI inference service. Run: uv run uvicorn predict:app --port 9696."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FEATURES = ["AT", "V", "AP", "RH"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = ROOT / "models" / "model.joblib"
    metadata_path = ROOT / "models" / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise RuntimeError("Model files missing. Run 'uv run python train.py' first.")
    metadata = json.loads(metadata_path.read_text())
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != metadata["model_sha256"]:
        raise RuntimeError("Model checksum does not match metadata.")
    if sklearn.__version__ != metadata["scikit_learn"]:
        raise RuntimeError("Use uv sync --locked: scikit-learn version differs from the training environment.")
    app.state.metadata = metadata
    app.state.model = joblib.load(model_path)
    yield


app = FastAPI(title="Power Output Estimator", version="1.0.0", lifespan=lifespan,
              description="Estimate same-hour full-load electrical output for the UCI CCPP plant. Not a temporal forecast or a plant-control system.")


class PlantConditions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    AT: float = Field(ge=-50, le=60, description="Ambient temperature in degrees Celsius", examples=[19.0])
    V: float = Field(ge=0, le=150, description="Exhaust vacuum in cm Hg", examples=[54.0])
    AP: float = Field(ge=800, le=1200, description="Ambient pressure in millibar", examples=[1013.0])
    RH: float = Field(ge=0, le=105, description="Relative humidity in percent; source has a few measurements slightly above 100", examples=[73.0])


class Prediction(BaseModel):
    net_output_mw: float
    model: str
    outside_training_range: bool
    warnings: list[str]


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(ROOT / "web" / "demo.html")


@app.get("/health")
def health(request: Request):
    metadata = request.app.state.metadata
    return {"status": "ok", "model": metadata["model_name"], "model_sha256": metadata["model_sha256"]}


@app.post("/predict", response_model=Prediction)
def predict(conditions: PlantConditions, request: Request):
    record = conditions.model_dump()
    metadata = request.app.state.metadata
    warnings = []
    for feature in FEATURES:
        bounds = metadata["training_ranges"][feature]
        if not bounds["min"] <= record[feature] <= bounds["max"]:
            warnings.append(f"{feature} is outside observed training values ({bounds['min']} to {bounds['max']}).")
    with threadpool_limits(limits=2):
        value = float(request.app.state.model.predict(pd.DataFrame([record], columns=FEATURES))[0])
    return Prediction(net_output_mw=round(value, 3), model=metadata["model_name"],
                      outside_training_range=bool(warnings), warnings=warnings)
