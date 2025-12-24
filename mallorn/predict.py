from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from typer import Typer

from .common import FEATS_DIR, MODELS_DIR, PREDS_DIR, mkdir, mksb, now
from .featurize import load_feats_df as _load_feats_df


typer = Typer()


@typer.command()
def predict(
    feats_dir: Path = FEATS_DIR,
    models_dir: Path = MODELS_DIR,
    preds_dir: Path = PREDS_DIR,
    top_k_percent: float = 5.0,  # Optimal strategy: Enforce ~5% positive rate
):
    """
    Generate predictions using the trained LightGBM model.
    Uses a Rank-Based Threshold (Top-K%) to handle class imbalance robustly.
    """
    with mksb() as sb:
        # 1. Load Data
        with sb.job("Loading test features"):
            test_feats_df = _load_feats_df(df_type="test", feats_dir=feats_dir)

            # Prepare X (Drop metadata)
            # Ensure these match the drop list in optimize.py
            ignore_cols = ["target", "object_id", "split", "English Translation", "SpecType"]
            X_test = test_feats_df.drop(columns=ignore_cols, errors="ignore")

            # Keep IDs for submission
            ids = test_feats_df.index if "object_id" not in test_feats_df.columns else test_feats_df["object_id"]

        # 2. Load Model
        model_path = models_dir / "lgb_final.txt"
        with sb.job(f"Loading LightGBM model from {model_path}"):
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}. Run `optimize.py train-best` first.")

            model = lgb.Booster(model_file=str(model_path))

        # 3. Predict Probabilities
        with sb.job("Generating probabilities"):
            y_probs = model.predict(X_test)

        # 4. Apply Rank-Based Thresholding (The "Optimal" Strategy)
        with sb.job(f"Applying Top-{top_k_percent}% Threshold Strategy"):
            # Calculate the threshold value that separates the top k%
            # np.percentile uses 0-100 scale, so we want the (100 - k)-th percentile
            threshold = np.percentile(y_probs, 100 - top_k_percent)

            y_pred = (y_probs >= threshold).astype(int)

            n_pos = y_pred.sum()
            print(f"\n   [Info] Top-{top_k_percent}% Threshold: {threshold:.6f}")
            print(f"   [Info] Positive Predictions: {n_pos}/{len(y_probs)} ({n_pos / len(y_probs):.2%})")

        # 5. Save Output
        out_file = preds_dir / f"predictions-{now()}.csv"
        with sb.job(f"Saving predictions to {out_file}"):
            preds_df = pd.DataFrame(
                {
                    "object_id": ids,
                    # "prob_tde": y_probs, # Useful for debugging/ensembling
                    "target": y_pred,
                }
            )

            # Ensure directory exists
            mkdir(preds_dir)

            # Save strictly as submission format (object_id, target) if needed,
            # but keeping prob_tde is helpful for analysis.
            # The competition likely expects 'object_id' and 'target'.
            preds_df.to_csv(out_file, index=False)
