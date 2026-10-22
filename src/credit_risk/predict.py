import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import add_features

logger = logging.getLogger(__name__)


def load_production(output_dir: Path) -> tuple[Any, float, str]:
    registry = json.loads((output_dir / "production.json").read_text(encoding="utf-8"))
    model_path = output_dir / registry["run_id"] / f"{registry['model']}.joblib"
    logger.info("Loading production model: run_id=%s path=%s", registry["run_id"], model_path)
    return joblib.load(model_path), float(registry["threshold"]), registry["run_id"]


def predict_records(records: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    logger.info("Predicting %d record(s)", len(records))
    model, threshold, run_id = load_production(output_dir)
    frame = add_features(pd.DataFrame(records))
    probabilities = model.predict_proba(frame)[:, 1]
    results = [{"bad_credit_probability": round(float(probability), 6),
             "decision": "review_or_decline" if probability >= threshold else "approve_or_review",
             "model_run_id": run_id} for probability in probabilities]
    logger.info("Completed predictions: count=%d model_run_id=%s", len(results), run_id)
    return results
