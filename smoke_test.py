"""Exercise the real HTTP service locally, in Docker, or at a supplied cloud URL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SAMPLE = {"AT": 19.0, "V": 54.0, "AP": 1013.0, "RH": 73.0}


def check_service(base_url: str):
    base_url = base_url.rstrip("/")
    with urlopen(base_url + "/health", timeout=30) as response:
        health = json.load(response)
    assert health["status"] == "ok"
    expected = json.loads((ROOT / "models" / "metadata.json").read_text())
    assert health["model_sha256"] == expected["model_sha256"]

    def post(record):
        request = Request(base_url + "/predict", data=json.dumps(record).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    prediction = post(SAMPLE)
    assert set(prediction) == {"net_output_mw", "model", "outside_training_range", "warnings"}
    assert isinstance(prediction["net_output_mw"], (int, float))
    assert 400 < prediction["net_output_mw"] < 520
    assert prediction["outside_training_range"] is False
    assert prediction == post(SAMPLE)
    try:
        post({"AT": 19.0})
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Incomplete input was accepted.")
    output = {"base_url": base_url, "health": health, "request": SAMPLE,
              "prediction": prediction, "deterministic_repeat": True, "incomplete_request_status": 422}
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:9696")
    args = parser.parse_args()
    check_service(args.base_url)
