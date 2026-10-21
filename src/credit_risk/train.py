from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from .config import RuntimeSettings, Settings, load_runtime_settings, write_json
from .data import load_dataset, split_data
from .evaluate import (
    choose_threshold,
    classification_metrics,
    cross_validated_metrics,
    permutation_importances,
    slice_metrics,
)
from .models import build_models


def train(
    settings: Settings,
    output_dir: Path,
    data_dir: Path,
    runtime: RuntimeSettings | None = None,
) -> Path:
    runtime = runtime or load_runtime_settings()
    frame, metadata = load_dataset(data_dir, runtime.dataset_url)
    x_train, x_test, y_train, y_test = split_data(frame, settings)
    runtime.reference_path.parent.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(runtime.reference_path, index=False)
    models = build_models(x_train, settings.random_state)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tracking_uri = runtime.mlflow_tracking_uri or f"sqlite:///{(output_dir.parent / 'mlflow.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-risk-training")
    results = {}
    with mlflow.start_run(run_name=run_id, tags={"project": "credit-risk-mlops", "stage": "comparison"}):
        mlflow.log_params({
            "random_state": settings.random_state,
            "test_size": settings.test_size,
            "cv_folds": settings.cv_folds,
            "false_negative_cost": settings.false_negative_cost,
            "false_positive_cost": settings.false_positive_cost,
            "dataset_rows": len(frame),
        })
        mlflow.log_dict(metadata, "dataset_metadata.json")
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True, tags={"model_family": name}):
                cv_metrics = cross_validated_metrics(model, x_train, y_train, settings.cv_folds, settings.random_state)
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(x_test)[:, 1]
                threshold = choose_threshold(y_test, probabilities, settings.false_negative_cost, settings.false_positive_cost)
                test_metrics = classification_metrics(y_test, probabilities, threshold)
                results[name] = {"cv": cv_metrics, "test": test_metrics, "threshold": threshold}
                mlflow.log_param("decision_threshold", threshold)
                mlflow.log_metrics(_flat_metrics("cv", cv_metrics) | _flat_metrics("test", test_metrics))
                model_path = run_dir / f"{name}.joblib"
                joblib.dump(model, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="models")
    winner = max(results, key=lambda name: results[name]["test"]["pr_auc"])
    winner_model = joblib.load(run_dir / f"{winner}.joblib")
    winner_threshold = results[winner]["threshold"]
    write_json(run_dir / "metrics.json", {"models": results, "winner": winner})
    write_json(run_dir / "config.json", settings.as_dict())
    write_json(run_dir / "data_profile.json", metadata | {"columns": list(frame.columns)})
    write_json(run_dir / "feature_importance.json", permutation_importances(winner_model, x_test, y_test, settings.random_state))
    write_json(run_dir / "slice_metrics.json", slice_metrics(winner_model, x_test, y_test, winner_threshold))
    (run_dir / "model_card.md").write_text(
        f"# Model card: {winner}\n\n"
        "Educational credit-risk classifier trained on the UCI German Credit dataset.\n\n"
        f"- Artifact: `{winner}.joblib`\n- Decision threshold: `{winner_threshold:.3f}`\n"
        "- Intended use: teaching end-to-end classical ML and MLOps.\n"
        "- Not intended for real lending, automated adverse action, or production use without validation.\n",
        encoding="utf-8",
    )
    with mlflow.start_run(run_name=run_id, tags={"project": "credit-risk-mlops", "stage": "registry"}):
        mlflow.log_params({"winner": winner, "decision_threshold": winner_threshold})
        mlflow.log_metrics(_flat_metrics("winner_test", results[winner]["test"]))
        model_info = mlflow.sklearn.log_model(
            winner_model,
            name="production_model",
            registered_model_name=runtime.mlflow_model_name,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "sklearn.tree._tree.Tree",
                "sklearn.utils._bunch.Bunch",
            ],
        )
        for artifact in ("metrics.json", "config.json", "data_profile.json",
                         "feature_importance.json", "slice_metrics.json", "model_card.md"):
            mlflow.log_artifact(str(run_dir / artifact), artifact_path="run_metadata")
        registered_version = _registered_version(model_info)
        MlflowClient(tracking_uri=mlflow.get_tracking_uri()).set_registered_model_alias(
            runtime.mlflow_model_name, runtime.mlflow_model_alias, registered_version
        )
    write_json(output_dir / "production.json", {
        "run_id": run_id,
        "model": winner,
        "threshold": winner_threshold,
        "mlflow_model": runtime.mlflow_model_name,
        "mlflow_alias": runtime.mlflow_model_alias,
        "mlflow_version": registered_version,
    })
    return run_dir


def _flat_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _registered_version(model_info) -> int:
    """Resolve the model version created by MLflow's registration call."""
    version = getattr(model_info, "registered_model_version", None)
    if version is not None:
        return int(version)
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    versions = client.search_model_versions(
        f"name = '{runtime.mlflow_model_name}' and run_id = '{mlflow.active_run().info.run_id}'"
    )
    if not versions:
        raise RuntimeError("MLflow registered the model without returning a model version")
    return max(int(item.version) for item in versions)


if __name__ == "__main__":
    runtime = load_runtime_settings()
    train(Settings(), runtime.artifact_dir, runtime.data_dir, runtime)
