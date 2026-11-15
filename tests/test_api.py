import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predict import app, FEATURES

SAMPLE = {"AT": 19.0, "V": 54.0, "AP": 1013.0, "RH": 73.0}
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_health_matches_artifact(client):
    response = client.get("/health")
    assert response.status_code == 200
    expected = json.loads((ROOT / "models/metadata.json").read_text())
    assert response.json()["status"] == "ok"
    assert response.json()["model_sha256"] == expected["model_sha256"]


def test_http_prediction_matches_saved_model(client):
    model = joblib.load(ROOT / "models/model.joblib")
    expected = float(model.predict(pd.DataFrame([SAMPLE], columns=FEATURES))[0])
    first = client.post("/predict", json=SAMPLE)
    assert first.status_code == 200
    assert abs(first.json()["net_output_mw"] - expected) <= 0.00051
    assert first.json() == client.post("/predict", json=SAMPLE).json()
    assert first.json()["outside_training_range"] is False


@pytest.mark.parametrize("record", [{"AT": 19.0}, {**SAMPLE, "unexpected": 1},
    {**SAMPLE, "RH": -1}, {**SAMPLE, "AT": "invalid"}, {**SAMPLE, "AP": 5000}])
def test_invalid_records_return_422(client, record):
    assert client.post("/predict", json=record).status_code == 422


def test_extrapolation_is_reported(client):
    result = client.post("/predict", json={**SAMPLE, "AT": 50.0})
    assert result.status_code == 200
    assert result.json()["outside_training_range"] is True
    assert result.json()["warnings"]


def test_demo_is_available(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Power Output" in response.text
