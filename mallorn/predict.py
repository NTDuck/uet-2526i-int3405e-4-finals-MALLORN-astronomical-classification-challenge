import json
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from typer import Typer

# Import common utilities consistent with your project structure
from .common import FEATS_DIR, MODELS_DIR, PREDS_DIR, SEED, mkdir
from .featurize import load_feats_df as _load_feats_df


typer = Typer()


@typer.command()
def predict(
    feats_dir: Path = FEATS_DIR,
    models_dir: Path = MODELS_DIR,
    submissions_dir: Path = PREDS_DIR,
    out_name: str = "submission.csv",
):
    """
    Retrains CatBoost on full data using best params (Focal Loss) and generates predictions.
    """
    # --- 1. Load Data ---
    print(">> Loading Data...")
    df_train = _load_feats_df(df_type="train", feats_dir=feats_dir)
    df_test = _load_feats_df(df_type="test", feats_dir=feats_dir)

    # --- 2. Prepare Feature Matrices ---
    # Drop non-feature columns
    drop_cols = ["target", "split", "SpecType", "English Translation"]

    # Prepare Train
    X_train = df_train.drop(columns=drop_cols, errors="ignore")
    y_train = df_train["target"]
    X_train = X_train.replace([np.inf, -np.inf], np.nan)

    # Prepare Test
    X_test = df_test.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # --- 3. Load Best Parameters and Threshold ---
    params_path = models_dir / "best_params_cb.json"
    τ_path = models_dir / "best_τ.json"

    if not params_path.exists() or not τ_path.exists():
        print("❌ Error: 'best_params_cb.json' or 'best_τ.json' not found.")
        print("   Please run 'opt-cb' and 'opt-τ' first.")
        return

    with open(params_path) as f:
        best_params = json.load(f)

    with open(τ_path) as f:
        τ_data = json.load(f)
        best_τ = τ_data["best_τ"]

    print(f"   Loaded Loss Function: {best_params.get('loss_function')}")
    print(f"   Loaded Threshold: {best_τ:.4f} (Strategy: {τ_data.get('strategy', 'unknown')})")

    # --- 4. Retrain on Full Dataset ---
    print(f">> Retraining on full dataset ({len(X_train)} samples)...")

    # Ensure consistency
    best_params["random_seed"] = SEED
    best_params["verbose"] = False

    # Initialize and Fit
    # CatBoost will automatically parse the "Focal:alpha=...;gamma=..." string in best_params
    model = cb.CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)

    # --- 5. Predict on Test ---
    print(">> Generating predictions...")
    probs = model.predict_proba(X_test)[:, 1]

    # Apply the optimized threshold
    preds = (probs >= best_τ).astype(int)

    # --- 6. Save Submission ---
    mkdir(submissions_dir)
    submission_path = submissions_dir / out_name

    # Create submission DataFrame
    # Assuming df_test.index contains the object/sample IDs
    submission = pd.DataFrame({"object_id": df_test.index, "target": preds})

    # Optional: Save probabilities to a separate file for potential ensembling later
    probs_df = pd.DataFrame({"object_id": df_test.index, "prob": probs})
    probs_df.to_csv(submissions_dir / "probs_cb.csv", index=False)

    # Save final submission
    submission.to_csv(submission_path, index=False)

    print(f"✅ Submission saved to: {submission_path}")
    print(f"   Positive Predictions: {preds.sum()} / {len(preds)} ({preds.mean():.2%})")


if __name__ == "__main__":
    typer()
