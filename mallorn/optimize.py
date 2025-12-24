import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from optuna_integration import LightGBMPruningCallback
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from typer import Typer

from .common import FEATS_DIR, MODELS_DIR, SEED, mkdir, mksb
from .featurize import load_feats_df


warnings.filterwarnings("ignore")

typer = Typer()


@typer.command()
def tune(n_trials: int | None = None, feats_dir: Path = FEATS_DIR):
    """
    Tune LightGBM with Grouped CV to prevent leakage from augmented clones.
    """
    with mksb() as sb:
        with sb.job("Loading Features"):
            df = load_feats_df("train", feats_dir)
            # Filter valid targets
            df = df[df["target"].notna()]

            # --- CRITICAL: EXTRACT GROUPS FOR CV ---
            # Augmented IDs look like "OriginalID_aug_0".
            # We must group by "OriginalID" to keep clones together.
            if "object_id" in df.columns:
                # Split by "_aug_" and take the first part to get original ID
                groups = df["object_id"].astype(str).apply(lambda x: x.split("_aug_")[0])
            else:
                # Fallback to index if object_id missing (should not happen based on featurize.py)
                groups = df.index.astype(str).to_series().apply(lambda x: x.split("_aug_")[0])

            # Drop non-feature columns
            drop_cols = ["target", "object_id", "split", "English Translation", "SpecType"]
            X = df.drop(columns=drop_cols, errors="ignore")
            y = df["target"].astype(int)

            print(f"   [Info] Shape: {X.shape}, Positive Rate: {y.mean():.2%}")
            print(f"   [Info] Unique Groups (Objects): {groups.nunique()}")

    def objective(trial):
        # 1. Hyperparameter Space (Restricted for Stability)
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": SEED,
            "n_estimators": 2000,
            # Prevent Overfitting: restrict leaf complexity
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            # Regularization is key for noisy astronomical data
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            # REMOVED scale_pos_weight: Data is already augmented/balanced.
            # Adding weight here creates double-bias.
            "scale_pos_weight": 1.0,
        }

        # 2. Grouped Cross-Validation (Prevents Leakage)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []

        # Pass 'groups' to split
        for train_idx, val_idx in sgkf.split(X, y, groups=groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)

            callbacks = [
                lgb.log_evaluation(period=0),
                # Pruning based on average_precision (more stable than F1)
                LightGBMPruningCallback(trial, "average_precision"),
            ]

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision", callbacks=callbacks)

            # Predict Probabilities
            probs = model.predict_proba(X_val)[:, 1]

            # --- DYNAMIC THRESHOLD OPTIMIZATION ---
            precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

            # F1 Calculation (safe divide)
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_curve = 2 * (precisions * recalls) / (precisions + recalls)
            f1_curve = np.nan_to_num(f1_curve)

            # If the model fails completely (all zeros), F1 is 0
            if len(f1_curve) == 0:
                best_f1 = 0.0
            else:
                best_f1 = np.max(f1_curve)

            f1_scores.append(best_f1)

        return np.mean(f1_scores)

    # Run Optimization
    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        direction="maximize",
        study_name="study-lgb-v5",  # New study name for clean slate
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("\nBest Trial:")
    print(f"  Value (F1): {study.best_value:.4f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save Best Params
    params_path = mkdir(MODELS_DIR) / "best_params_lgb.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Saved best params to {params_path}")


@typer.command()
def train(feats_dir: Path = FEATS_DIR, models_dir: Path = MODELS_DIR):
    """Train final model on ALL data using best params."""
    params_path = models_dir / "best_params_lgb.json"
    if not params_path.exists():
        print("Run `tune` first!")
        return

    with open(params_path) as f:
        params = json.load(f)

    params.update(
        {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "random_state": SEED,
            "scale_pos_weight": 1.0,  # Ensure consistency
        }
    )

    df = load_feats_df("train", feats_dir)
    df = df[df["target"].notna()]

    drop_cols = ["target", "object_id", "split", "English Translation", "SpecType"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["target"].astype(int)

    print("Training final model on full dataset...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    model_path = models_dir / "lgb_final.txt"
    model.booster_.save_model(model_path)
    print(f"Saved model to {model_path}")
