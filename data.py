"""Retrieve one worksheet of the public UCI CCPP data and prepare fixed splits."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "power_plant.csv"
MANIFEST_PATH = ROOT / "data" / "source_manifest.json"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
SOURCE_SHA256 = "cc7b2a4977c0a44e8221c91d9a7e5746b3c68186cff7e5c61c70af6432b98c7a"
FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_data() -> Path:
    """Convert Sheet1 only; other worksheets are reshufflings, not extra samples."""
    with urlopen(SOURCE_URL, timeout=90) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("UCI source archive checksum changed; inspect before proceeding.")
    with ZipFile(io.BytesIO(content)) as archive:
        workbook = io.BytesIO(archive.read("CCPP/Folds5x2_pp.xlsx"))
    frame = pd.read_excel(workbook, sheet_name="Sheet1", engine="openpyxl")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[FEATURES + [TARGET]].to_csv(DATA_PATH, index=False, lineterminator="\n")
    manifest = {
        "dataset": "Combined Cycle Power Plant",
        "source_url": SOURCE_URL,
        "source_page": "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
        "doi": "10.24432/C5002N",
        "creators": "Pinar Tufekci and Heysem Kaya",
        "license": "CC BY 4.0",
        "source_archive_sha256": SOURCE_SHA256,
        "worksheet": "Sheet1",
        "csv_sha256": sha256(DATA_PATH),
        "rows_before_deduplication": len(frame),
        "conversion": "pandas read_excel Sheet1 then to_csv(index=False, lineterminator='\\n')",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return DATA_PATH


def load_data() -> tuple[pd.DataFrame, dict]:
    if not DATA_PATH.exists():
        download_data()
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Missing data/source_manifest.json; run python data.py to restore the source.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256(DATA_PATH) != manifest["csv_sha256"]:
        raise ValueError("CSV checksum mismatch; run python data.py to restore canonical bytes.")
    raw = pd.read_csv(DATA_PATH)
    if raw.columns.tolist() != FEATURES + [TARGET]:
        raise ValueError("Unexpected data columns")
    if raw[TARGET].isna().any() or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Unexpected missing or non-finite source values")
    frame = raw.drop_duplicates().reset_index(drop=True)
    # After exact duplicates are removed, no repeated feature vector may cross splits.
    if frame.duplicated(FEATURES).any():
        raise ValueError("Repeated feature vectors detected; revise splitting to use groups.")
    quality = {
        "source_rows": len(raw), "clean_rows": len(frame),
        "duplicates_removed": len(raw) - len(frame),
        "missing_values": raw.isna().sum().to_dict(),
        "csv_sha256": manifest["csv_sha256"],
    }
    return frame, quality


def split_data(frame: pd.DataFrame):
    full_train, test = train_test_split(frame, test_size=0.2, random_state=SEED)
    train, validation = train_test_split(full_train, test_size=0.25, random_state=SEED)
    feature_sets = [set(map(tuple, part[FEATURES].to_numpy())) for part in [train, validation, test]]
    assert feature_sets[0].isdisjoint(feature_sets[1])
    assert feature_sets[0].isdisjoint(feature_sets[2])
    assert feature_sets[1].isdisjoint(feature_sets[2])
    return train.copy(), validation.copy(), test.copy()


if __name__ == "__main__":
    print("Saved:", download_data())
    _, quality = load_data()
    print(json.dumps(quality, indent=2))
