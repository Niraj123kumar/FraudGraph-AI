import json

import numpy as np

from src.ml.evaluate import _threshold_curve_points, run_evaluation


def test_missing_inputs_persist_an_unavailable_report(tmp_path):
    report_path = tmp_path / "evaluation_report.json"
    report = run_evaluation(
        data_path=tmp_path / "missing.csv",
        model_path=tmp_path / "missing.joblib",
        metadata_path=tmp_path / "missing.json",
        report_path=report_path,
    )

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == persisted
    assert report["available"] is False
    assert "Dataset is not available" in report["reason"]
    assert report["metrics"] is None
    assert report["curves"] is None


def test_threshold_curves_keep_endpoints_and_model_cutoff():
    labels = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.7, 0.9])

    sampled = _threshold_curve_points(labels, probabilities, 0.65, max_points=10)
    roc = sampled["roc"]
    pr = sampled["precision_recall"]

    assert len(roc["false_positive_rate"]) <= 10
    assert len(roc["false_positive_rate"]) == len(roc["true_positive_rate"])
    assert roc["false_positive_rate"][0] == roc["true_positive_rate"][0] == 0
    assert roc["false_positive_rate"][-1] == roc["true_positive_rate"][-1] == 1
    assert pr["recall"][0] == 0 and pr["recall"][-1] == 1
    assert any(np.isclose(recall, 2 / 3) and np.isclose(precision, 1) for recall, precision in zip(pr["recall"], pr["precision"]))

