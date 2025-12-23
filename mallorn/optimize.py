import json
import re
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from typer import Typer

from .common import FEATS_DIR, MODELS_DIR, SEED, mkdir
from .featurize import load_feats_df as _load_feats_df


typer = Typer()


def _get_root_id(obj_id: str) -> str:
    """Removes _aug_X suffix to get the original object ID."""
    return re.sub(r"_aug_\d+$", "", str(obj_id))


def _get_clean_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED):
    """
    Creates folds based on ORIGINAL objects only.
    Ensures Train gets (Originals + Clones), Validation gets (Originals only).
    """
    df = df.copy()
    df["root_id"] = df.index.map(_get_root_id)
    df["is_aug"] = df.index.str.contains("_aug_")

    originals = df[~df["is_aug"]]
    if originals.empty:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for t, v in skf.split(df, df["target"]):
            yield t, v
        return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx_orig, val_idx_orig in skf.split(originals, originals["target"]):
        train_roots = set(originals.iloc[train_idx_orig]["root_id"])
        val_roots = set(originals.iloc[val_idx_orig]["root_id"])

        train_mask = df["root_id"].isin(train_roots)
        val_mask = df["root_id"].isin(val_roots) & (~df["is_aug"])

        yield np.where(train_mask)[0], np.where(val_mask)[0]


@typer.command(name="opt-cb")
def opt_cb(
    n_trials: int = 50,
    feats_dir: Path = FEATS_DIR,
    models_dir: Path = MODELS_DIR,
):
    """Optimize CatBoost Hyperparameters with Focal Loss and Leakage-Free Validation."""
    df = _load_feats_df(df_type="train", feats_dir=feats_dir)

    X = df.drop(columns=["target", "split", "SpecType", "English Translation"], errors="ignore")
    y = df["target"]

    X = X.replace([np.inf, -np.inf], np.nan)

    print(f"Optimization Start: {len(df)} samples ({y.sum()} positives)")
    print("Using Focal Loss (No LogLoss/ScalePosWeight) with Leakage-Free Validation")

    def objective(trial):
        # 1. Suggest Focal Loss specific parameters
        # alpha: balancing parameter (similar to scale_pos_weight inverse).
        # Lower alpha downweights negatives. range [0, 1].
        focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.7)

        # gamma: focusing parameter. Higher gamma focuses more on hard examples.
        focal_gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)

        # 2. Other Hyperparameters
        # Note: Removed scale_pos_weight as Focal Loss handles imbalance via alpha
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            # Construct the CatBoost specific string for Focal Loss
            "loss_function": f"Focal:focal_alpha={focal_alpha};focal_gamma={focal_gamma}",
            "eval_metric": "PRAUC",
            "random_seed": SEED,
            "verbose": False,
            "allow_writing_files": False,
        }

        f1_scores = []

        for train_idx, val_idx in _get_clean_splits(df, n_splits=5):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                # early_stopping_rounds=50,
                verbose=False,
            )

            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds)
            f1_scores.append(f1)

        return np.mean(f1_scores)

    study = optuna.create_study(storage="sqlite:///optuna.db", direction="maximize", study_name="cb-study-v5", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)  # pyright: ignore[reportArgumentType]

    print("Best params:", study.best_params)

    # Reconstruct the full params dictionary including the constructed loss string
    best_params = study.best_params.copy()

    # Extract alpha/gamma to build the string, then remove them as raw keys
    # (CatBoost doesn't accept 'focal_alpha' as a direct kwarg, only inside loss_function string)
    f_alpha = best_params.pop("focal_alpha")
    f_gamma = best_params.pop("focal_gamma")

    final_params = {
        **best_params,
        "loss_function": f"Focal:focal_alpha={f_alpha};focal_gamma={f_gamma}",
        "eval_metric": "F1",
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
    }

    mkdir(models_dir)
    with open(models_dir / "best_params_cb.json", "w") as f:
        json.dump(final_params, f, indent=4)


@typer.command(name="opt-τ")
def opt_τ(
    feats_dir: Path = FEATS_DIR,
    models_dir: Path = MODELS_DIR,
):
    """Optimize Threshold (τ) using Rank-Based Strategy."""
    df = _load_feats_df(df_type="train", feats_dir=feats_dir)

    X = df.drop(columns=["target", "split", "SpecType", "English Translation"], errors="ignore").replace([np.inf, -np.inf], np.nan)
    y = df["target"]

    # Load best params (now contains the Focal Loss string)
    with open(models_dir / "best_params_cb.json") as f:
        params = json.load(f)
        print(f"Loaded Params with Loss: {params.get('loss_function')}")

    oof_probs = np.zeros(len(df))
    valid_mask_all = np.zeros(len(df), dtype=bool)

    print("Training for Threshold Optimization...")
    for train_idx, val_idx in _get_clean_splits(df, n_splits=20):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        valid_mask_all[val_idx] = True

    y_real = y[valid_mask_all]
    probs_real = oof_probs[valid_mask_all]

    # Strategy 1: Standard F1 Maximization
    precisions, recalls, thresholds = precision_recall_curve(y_real, probs_real)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_τ_f1 = thresholds[best_idx]
    best_score_f1 = f1_scores[best_idx]

    # Strategy 2: Rank-Based (Top 5%)
    target_ratio = 0.05
    k = int(len(probs_real) * target_ratio)
    sorted_probs = np.sort(probs_real)[::-1]
    best_τ_rank = sorted_probs[k]

    y_pred_rank = (probs_real >= best_τ_rank).astype(int)
    best_score_rank = f1_score(y_real, y_pred_rank)

    print("\n--- Threshold Optimization Results ---")
    print(f"1. Max F1 Threshold: {best_τ_f1:.4f} (CV F1: {best_score_f1:.4f})")
    print(f"2. Rank-Based (Top 5%): {best_τ_rank:.4f} (CV F1: {best_score_rank:.4f})")

    final_τ = best_τ_rank if best_score_rank > (best_score_f1 - 0.02) else best_τ_f1

    print(f"\n>> Selected Threshold: {final_τ:.4f}")

    with open(models_dir / "best_τ.json", "w") as f:
        json.dump({"best_τ": float(final_τ), "strategy": "rank" if final_τ == best_τ_rank else "f1"}, f)
