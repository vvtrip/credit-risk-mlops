import pandas as pd
import pytest

from credit_risk.monitor import drift_report


def test_drift_report_detects_numeric_shift():
    reference = pd.DataFrame({"age": [20, 21, 22, 23, 24, 25]})
    current = pd.DataFrame({"age": [70, 71, 72, 73, 74, 75]})
    assert drift_report(reference, current)["age"]["drifted"]


def test_drift_report_covers_categorical_columns():
    reference = pd.DataFrame({"housing": ["own", "own", "rent"]})
    current = pd.DataFrame({"housing": ["free", "free", "free"]})
    result = drift_report(reference, current)
    assert result["housing"]["metric"] == "js_divergence"


def test_drift_report_uses_configured_categorical_threshold():
    reference = pd.DataFrame({"housing": ["own", "own", "rent"]})
    current = pd.DataFrame({"housing": ["free", "free", "free"]})
    result = drift_report(reference, current, js_threshold=10.0)
    assert not result["housing"]["drifted"]


def test_monitor_cli_reports_missing_files_clearly():
    from credit_risk.cli import main

    with pytest.raises(FileNotFoundError, match="Monitoring file"):
        import sys
        original = sys.argv
        sys.argv = ["credit-risk", "monitor", "--input", "missing.csv"]
        try:
            main()
        finally:
            sys.argv = original
