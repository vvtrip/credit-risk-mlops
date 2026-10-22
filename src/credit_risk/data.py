import hashlib
import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import certifi
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Settings, load_runtime_settings
from .features import add_features

RAW_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "dependents", "telephone", "foreign_worker", "target",
]

logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path, dataset_url: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_url = dataset_url or load_runtime_settings().dataset_url
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "german.data"
    if not archive.exists():
        logger.info("Dataset not found; downloading from %s", dataset_url)
        zip_path = data_dir / "german.zip"
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(dataset_url, context=context) as response:
            zip_path.write_bytes(response.read())
        import zipfile
        with zipfile.ZipFile(zip_path) as zipped:
            member = next(name for name in zipped.namelist() if name.endswith("german.data"))
            archive.write_bytes(zipped.read(member))
        logger.info("Downloaded dataset to %s", archive)
    else:
        logger.info("Using cached dataset: %s", archive)
    frame = pd.read_csv(archive, sep=r"\s+", header=None, names=RAW_COLUMNS)
    frame["target"] = frame["target"].map({1: 0, 2: 1}).astype("int8")
    metadata = {
        "source_url": dataset_url,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "rows": len(frame),
        "columns": len(frame.columns),
        "positive_label": "bad credit (original label 2)",
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validate_dataset(frame)
    logger.info("Validated dataset: rows=%d", len(frame))
    return frame, metadata


def validate_dataset(frame: pd.DataFrame) -> None:
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame["target"].isna().any() or not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 (good) and 1 (bad)")
    if len(frame) < 100:
        raise ValueError("Dataset is unexpectedly small")


def split_data(frame: pd.DataFrame, settings: Settings):
    features = add_features(frame.drop(columns=["target"]))
    return train_test_split(
        features, frame["target"], test_size=settings.test_size,
        stratify=frame["target"], random_state=settings.random_state,
    )


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = frame.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )
