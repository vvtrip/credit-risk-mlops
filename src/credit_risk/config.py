import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def configure_logging() -> None:
    """Configure consistent console logging for CLI and module entry points."""
    level_name = os.getenv("CREDIT_RISK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise TypeError(f"Invalid CREDIT_RISK_LOG_LEVEL: {level_name}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    positive_label: int = 1
    false_negative_cost: float = 5.0
    false_positive_cost: float = 1.0
    drift_psi_threshold: float = 0.2
    drift_js_threshold: float = 0.1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSettings:
    artifact_dir: Path = Path("artifacts")
    data_dir: Path = Path("data/raw")
    reference_path: Path = Path("data/reference.csv")
    mlflow_tracking_uri: str | None = None
    mlflow_model_name: str = "credit-risk-production"
    mlflow_model_alias: str = "champion"
    dataset_url: str = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"


def load_runtime_settings() -> RuntimeSettings:
    values: dict[str, Any] = {}
    path_values = {
        "artifact_dir": "CREDIT_RISK_ARTIFACT_DIR",
        "data_dir": "CREDIT_RISK_DATA_DIR",
        "reference_path": "CREDIT_RISK_REFERENCE_PATH",
        "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
        "mlflow_model_name": "MLFLOW_MODEL_NAME",
        "mlflow_model_alias": "MLFLOW_MODEL_ALIAS",
        "dataset_url": "CREDIT_RISK_DATASET_URL",
    }
    for field, environment_name in path_values.items():
        value = os.getenv(environment_name)
        if value:
            values[field] = Path(value) if field.endswith(("_dir", "_path")) else value
    return RuntimeSettings(**values)


def load_settings(path: Path | None = None) -> Settings:
    if path is None:
        return Settings()
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return Settings(**values)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
