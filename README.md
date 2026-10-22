# Credit Risk MLOps: An End-to-End Classical ML Project

This repository is a pedagogical, production-shaped credit-risk project. It predicts whether an applicant is a **good** or **bad** credit risk using the public [UCI Statlog German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

It is deliberately **classical machine learning only**: no neural networks or deep learning. The project demonstrates:

- Reproducible data acquisition, validation, and feature engineering
- A leakage-safe scikit-learn preprocessing pipeline
- Bagging (`RandomForestClassifier`)
- Boosting (`HistGradientBoostingClassifier`)
- Model stacking (`StackingClassifier`)
- Cross-validation, threshold selection, and business metrics
- Permutation importance and fairness-oriented slice checks
- Versioned model artifacts and metadata
- FastAPI inference service and a batch prediction CLI
- Data/model/feature drift monitoring
- Docker, CI, tests, experiment tracking, and configuration

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m credit_risk train
python -m credit_risk predict --input examples/request.json
python -m credit_risk monitor --input examples/monitor.csv
python -m uvicorn --app-dir src credit_risk.api:app --reload
```

The first training run downloads the dataset into `data/raw/`. Data and generated models are ignored by git.
Run training before monitoring so that `data/reference.csv` exists:

```powershell
python -m credit_risk train --config configs/default.yaml
python -m credit_risk monitor --input examples/monitor.csv
```

Runtime defaults are centralized in [config.py](C:/Users/Vaibhav/Downloads/credit-risk-mlops/src/credit_risk/config.py). Copy `.env.example` to `.env` and change values for another environment. The CLI and API load `CREDIT_RISK_*` and `MLFLOW_*` settings from `.env` or the process environment; command-line flags still take precedence for training output, data, and monitoring input paths. Do not commit secrets to `.env`.
CLI commands emit timestamped lifecycle logs to stderr. Set `CREDIT_RISK_LOG_LEVEL=DEBUG`, `INFO` (the default), `WARNING`, or `ERROR` to control verbosity.

Open the browser dashboard at [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app). It provides a responsive application form, API status indicator, sample-data shortcut, risk probability meter, model run ID, and threshold-aware action band. Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Training also writes an MLflow SQLite tracking store to `mlflow.db`. Open the experiment portal in a second terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000). The `credit-risk-training` experiment shows the parent training run, nested baseline/bagging/boosting/stacking runs, parameters, CV/test metrics, thresholds, model artifacts, dataset metadata, and the selected winner. The registered model `credit-risk-production` uses the `champion` alias for the latest successful winner. This is a local tracking portal for teaching, not a hosted production registry.

### Run the complete local stack with Docker Compose

After installing Docker Desktop, train and start the API/UI plus MLflow portal with:

```powershell
docker compose --profile training run --rm trainer
docker compose up --build -d api mlflow
```

Open:

- [Credit Risk Studio](http://127.0.0.1:8000/app)
- [FastAPI docs](http://127.0.0.1:8000/docs)
- [MLflow portal](http://127.0.0.1:5000)

The `trainer` profile downloads the dataset and writes model artifacts to the host `artifacts/` directory. The API mounts those artifacts read-only, while MLflow and the trainer share `mlflow.db` and `mlruns/`. To retrain with the same stack, run the trainer command again; it creates a new MLflow run and registered model version, then restart only the API if you want to force a fresh container:

Compose ports can be changed without editing YAML:

```powershell
$env:API_PORT = "8080"
$env:MLFLOW_PORT = "5050"
docker compose up --build -d api mlflow
```

```powershell
docker compose --profile training run --rm trainer
docker compose restart api
```

Stop the services with:

```powershell
docker compose down
```

## Project map

```text
src/credit_risk/
  config.py       Typed settings and paths
  data.py         Download, schema checks, split, preprocessing
  features.py     Domain feature engineering
  models.py       Baselines, bagging, boosting, stacking
  evaluate.py     Metrics, threshold selection, explainability
  train.py        End-to-end training entry point
  predict.py      Batch inference
  monitor.py      Feature drift checks
  api.py          FastAPI online inference
tests/            Unit and integration tests
configs/          Reproducible experiment configuration
examples/         Sample online and batch requests
```

## End-to-end lifecycle

1. **Frame the problem:** missing a bad-credit applicant is costlier than declining a good one, so the default threshold is selected with an explicit FN/FP cost matrix.
2. **Acquire and validate data:** `credit_risk.data` downloads the UCI source, verifies the expected 20 features plus target, and records dataset metadata.
3. **Split correctly:** a stratified train/test split is created before fitting any preprocessing.
4. **Prepare features:** categorical columns are one-hot encoded; numerical columns are median-imputed and scaled. All transformations are inside a pipeline to prevent leakage.
5. **Compare model families:** logistic regression is a transparent baseline; random forest is bagging; histogram gradient boosting is boosting; stacking combines diverse base learners using out-of-fold predictions.
6. **Evaluate:** cross-validation, ROC-AUC, PR-AUC, recall, precision, Brier score, confusion matrix, and cost-sensitive threshold selection are reported.
7. **Explain and slice:** permutation importance and group-level metrics by age band and gender are emitted for review. These are diagnostic checks, not claims of fairness.
8. **Package:** the selected pipeline, metrics, configuration, feature schema, and run ID are saved under `artifacts/<run_id>/`.
9. **Serve:** the same serialized pipeline powers the API and batch CLI, avoiding training/serving skew.
10. **Monitor:** feature drift is checked with PSI for numeric features and Jensen-Shannon divergence for categorical features. Label-based prediction-quality monitoring is a future extension.

## MLOps practices included

- **Reproducibility:** pinned dependency ranges, a seed, configuration snapshot, dataset checksum, and run ID
- **Data validation:** required columns, allowed target values, missingness and type checks
- **Experiment tracking:** every run writes `metrics.json`, `config.json`, `data_profile.json`, and `model_card.md`
- **MLflow portal:** training logs parameters, metrics, nested model comparisons, dataset metadata, serialized models, and the selected winner to `mlflow.db`
- **Model registry:** every successful training run registers the selected winner as a new version of `credit-risk-production`, moves the `champion` alias to that version, and records the version in `artifacts/production.json`
- **CI:** lint and unit tests in GitHub Actions
- **Serving:** health endpoint, request validation, structured responses, and model version in every prediction
- **User interface:** a dependency-free browser dashboard served by FastAPI and backed by the same `/predict` contract
- **Monitoring:** numeric/categorical drift report and configurable PSI/JS thresholds suitable for a scheduled batch job
- **Governance:** model card, limitations, ethical considerations, and an explicit warning that this is educational, not a lending decision system

## What is the MLOps component?

The machine-learning component is the feature engineering, preprocessing, model training, ensemble comparison, threshold selection, and evaluation code. The MLOps component is the operational system around that model:

- **Reproducibility:** `pyproject.toml`, configuration in `configs/default.yaml`, fixed random seeds, dataset checksums, and versioned run directories
- **Environment portability:** runtime paths, dataset source, MLflow URI/model identifiers, and Compose ports are environment-configurable with `.env`
- **Data operations:** automatic dataset download/cache, schema validation, metadata capture, train/test splitting, and a reference population for monitoring
- **Experiment management:** each training run stores metrics, configuration, data profile, feature importance, slice metrics, model card, and serialized models in `artifacts/<run_id>/`
- **Model registry/promotion:** MLflow registers the selected model as `credit-risk-production`; `artifacts/production.json` identifies the local serving artifact, threshold, MLflow version, and `champion` alias
- **Testing and CI:** pytest tests, Ruff linting, Python package configuration, and GitHub Actions
- **Deployment:** Docker packaging, FastAPI health/readiness endpoints, request validation, API documentation, and batch inference
- **Compose orchestration:** `compose.yaml` runs the API, MLflow portal, and an optional training job with shared, persistent local artifacts
- **Monitoring:** PSI for numeric drift, Jensen-Shannon divergence for categorical drift, and configurable alert thresholds
- **Governance:** model cards, limitations, fairness-oriented slices, cost-sensitive decisions, and an explicit educational-use warning

In a larger production system, these local files would typically be replaced or extended by managed artifact storage, scheduled workflows, a feature store, secrets management, centralized logs/metrics, alerting, access control, and formal model approval.

## Dataset and limitations

The German Credit dataset is small, old, and contains encoded categorical values. Its labels and demographic variables should not be used for real lending decisions. The project intentionally keeps the original variables visible for teaching, but reports slice metrics and calls out that fairness requires domain, legal, and societal review beyond a metric.

## Useful commands

```powershell
python -m credit_risk train --config configs/default.yaml
python -m credit_risk train --config configs/default.yaml --output-dir artifacts
python -m credit_risk predict --input data/new_applications.csv --output data/predictions.csv
python -m pytest -q
python -m ruff check .
```

## Next extensions

For a real deployment, add a feature store, managed artifact storage, authenticated service-to-service calls, secrets management, a scheduled monitor, label-based model-quality monitoring, a human review queue, immutable audit logs, and a formal validation process.
