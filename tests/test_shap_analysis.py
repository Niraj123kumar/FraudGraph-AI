import json

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from scripts.shap_analysis import run_analysis
from src.data.preprocessing import EXPECTED_COLS, FEATURE_COLS


def test_shap_analysis_writes_all_artifacts(tmp_path):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(40, len(EXPECTED_COLS))), columns=EXPECTED_COLS)
    frame["Class"] = np.array([0, 1] * 20)
    model = DecisionTreeClassifier(max_depth=2, random_state=3).fit(frame[FEATURE_COLS], frame["Class"])
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(model, models / "transaction_model.joblib")
    (models / "model_metadata.json").write_text(json.dumps({"feature_names": FEATURE_COLS}), encoding="utf-8")
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)

    paths = run_analysis(models, data, tmp_path / "shap", sample_size=10)

    assert all(path.is_file() for path in paths.values())
    assert json.loads(paths["mean_absolute"].read_text()) 
    values = np.load(paths["values"])
    assert values["values"].shape == (10, len(FEATURE_COLS))
    assert values["labels"].sum() > 0
