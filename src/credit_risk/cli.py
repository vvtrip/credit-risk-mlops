import argparse
import json
from pathlib import Path

from .config import configure_logging, load_runtime_settings, load_settings
from .monitor import drift_report
from .predict import predict_records
from .train import train


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Credit-risk ML lifecycle CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--config", type=Path)
    runtime = load_runtime_settings()
    training.add_argument("--output-dir", type=Path, default=runtime.artifact_dir)
    training.add_argument("--data-dir", type=Path, default=runtime.data_dir)
    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--output", type=Path)
    prediction.add_argument("--output-dir", type=Path, default=runtime.artifact_dir)
    monitoring = subparsers.add_parser("monitor")
    monitoring.add_argument("--reference", type=Path, default=runtime.reference_path)
    monitoring.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        print(train(load_settings(args.config), args.output_dir, args.data_dir, runtime))
    elif args.command == "predict":
        result = predict_records(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    else:
        import pandas as pd
        missing = [str(path) for path in (args.reference, args.input) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Monitoring file(s) not found: "
                + ", ".join(missing)
                + ". Run `python -m credit_risk train` first to create the reference file."
            )
        reference = pd.read_csv(args.reference)
        current = pd.read_csv(args.input)
        report = drift_report(reference, current)
        if not report:
            raise ValueError(
                "No columns overlap between the reference and current monitoring files."
            )
        print(json.dumps(report, indent=2))
