"""Generate offline SHAP artifacts for a saved fraud model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import FEATURE_COLS, TARGET_COL


def run_analysis(models_dir: Path, data_path: Path, out: Path, sample_size: int = 5000) -> dict[str, Path]:
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names", FEATURE_COLS)
    if feature_names != FEATURE_COLS:
        raise ValueError("Saved model feature schema does not match the dataset schema")
    model = joblib.load(models_dir / "transaction_model.joblib")
    frame = pd.read_csv(data_path, usecols=[*FEATURE_COLS, TARGET_COL])
    sample_size = min(sample_size, len(frame))
    if sample_size < 2:
        raise ValueError("The dataset must contain at least two rows")
    rng = np.random.default_rng(42)
    fraud = np.flatnonzero(frame[TARGET_COL].to_numpy() == 1)
    legitimate = np.flatnonzero(frame[TARGET_COL].to_numpy() == 0)
    fraud_count = min(len(fraud), max(1, round(sample_size * len(fraud) / len(frame))))
    legitimate_count = sample_size - fraud_count
    if legitimate_count > len(legitimate):
        legitimate_count = len(legitimate)
        fraud_count = sample_size - legitimate_count
    selected = np.concatenate((
        rng.choice(fraud, size=fraud_count, replace=False),
        rng.choice(legitimate, size=legitimate_count, replace=False),
    ))
    selected.sort()
    X = frame.iloc[selected][FEATURE_COLS]

    import shap

    values = np.asarray(shap.TreeExplainer(model)(X).values)
    if values.ndim == 3:
        values = values[:, :, -1]
    mean_absolute = dict(zip(FEATURE_COLS, np.abs(values).mean(axis=0).tolist()))
    out.parent.mkdir(parents=True, exist_ok=True)
    png_path = out.with_name(f"{out.name}_summary.png")
    json_path = out.with_name(f"{out.name}_mean_abs.json")
    npz_path = out.with_name(f"{out.name}_values.npz")
    shap.summary_plot(values, X, show=False, plot_type="bar")
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    json_path.write_text(json.dumps(mean_absolute, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, values=values, features=X.to_numpy(), labels=frame.iloc[selected][TARGET_COL].to_numpy())
    return {"summary": png_path, "mean_absolute": json_path, "values": npz_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()
    run_analysis(args.models_dir, args.data, args.out, args.sample_size)


if __name__ == "__main__":
    main()
