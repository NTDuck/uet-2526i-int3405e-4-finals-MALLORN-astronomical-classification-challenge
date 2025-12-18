import json
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from typer import Typer

from .common import FEATS_DIR, MODELS_DIR, SEED, mkdir, mksb, now
from .featurize import load_feats_df as _load_feats_df


typer = Typer()


@typer.command(name="opt-cb")
def opt_cb(n_trials: int | None = None, feats_dir: Path = FEATS_DIR):
    with mksb() as sb:
        with sb.job("Loading featurized dataset"):
            feats_df = _load_feats_df(df_type="train", feats_dir=feats_dir)
            X = feats_df.drop(columns=["target"])
            y = feats_df["target"]

            # Calculate imbalance ratio for scale_pos_weight
            N_TDES = y.sum()
            N_NON_TDES = len(y) - N_TDES
            base_scale = N_NON_TDES / N_TDES if N_TDES > 0 else 1.0

    def objective(trial: optuna.Trial):
        loss_type = trial.suggest_categorical("loss_type", ["Logloss", "Focal"])

        params = {
            # Tree Structure
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "depth": trial.suggest_int("depth", 4, 10),
            # Learning Dynamics
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            # Regularization (Crucial for high-dim spectral data)
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            # Rare Event Handling
            # Prevent the model from "averaging out" the few TDEs in a leaf.
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            # Fixed
            "task_type": "GPU",
            "eval_metric": "PRAUC",
            "verbose": False,
            "random_seed": SEED,
            "allow_writing_files": False,
            "thread_count": 4,
        }

        if loss_type == "Logloss":
            params["loss_function"] = "Logloss"
            w_mult = trial.suggest_float("weight_multiplier", 0.5, 3.0)
            params["scale_pos_weight"] = base_scale * w_mult
        else:  # if loss_type == "Focal"
            alpha = trial.suggest_float("focal_alpha", 0.1, 0.9)
            gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
            params["loss_function"] = f"Focal:focal_alpha={alpha};focal_gamma={gamma}"

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        prauc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

            pred_probs = model.predict_proba(X_val)[:, 1]
            prauc_scores.append(average_precision_score(y_val, pred_probs))

        return np.mean(prauc_scores)

    study = optuna.create_study(storage="sqlite:///optuna.db", direction="maximize", study_name="cb_study_v2", load_if_exists=True)
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)  # pyright: ignore[reportArgumentType]
    except KeyboardInterrupt:
        print(f"Catboost optimized with best AUC: {study.best_value:.6f}")


@typer.command(name="opt-τ")
def opt_τ(feats_dir: Path = FEATS_DIR, models_dir: Path = MODELS_DIR):
    with mksb() as sb:
        with sb.job("Loading featurized dataset"):
            train_feats_df = _load_feats_df(df_type="train", feats_dir=feats_dir)
            X = train_feats_df.drop(columns=["target"])
            y = train_feats_df["target"]

            study = optuna.load_study(storage="sqlite:///optuna.db", study_name="cb_study_v2")

            cb_params = study.best_params

            loss_type = cb_params.pop("loss_type", "Logloss")
            if loss_type == "Logloss":
                cb_params["loss_function"] = "Logloss"
                w_mult = cb_params.pop("weight_multiplier", 1.0)

                # Calculate base scale again to be safe
                n_pos = y.sum()
                n_neg = len(y) - n_pos
                base_scale = n_neg / n_pos if n_pos > 0 else 1.0

                cb_params["scale_pos_weight"] = base_scale * w_mult
            else:
                alpha = cb_params.pop("focal_alpha")
                gamma = cb_params.pop("focal_gamma")
                cb_params["loss_function"] = f"Focal:focal_alpha={alpha};focal_gamma={gamma}"

            cb_params.update({
                "task_type": "GPU",
                "eval_metric": "PRAUC",
                "verbose": False,
                "random_seed": SEED,
                "allow_writing_files": False,
                "thread_count": 4,
            })  # fmt: skip

        with sb.job("Populating OOF predictions"):
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

            oof_preds = np.zeros(len(X))

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                cb_model = cb.CatBoostClassifier(**cb_params)
                cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

                # Save model for prediction phase
                cb_model.save_model(mkdir(models_dir / "catboost") / f"model-fold-{fold}-{now()}.cbm")

                oof_preds[val_idx] = cb_model.predict_proba(X_val)[:, 1]

        with sb.job("Optimizing Threshold τ for F1"):
            best_τ, best_f1 = _opt_τ_and_f1(y, oof_preds)

            with open(mkdir(models_dir / "τ") / f"{now()}.json", "w") as file:
                json.dump({
                    "best_τ": best_τ,
                    "best_f1": best_f1,
                }, file, indent=4)  # fmt: skip

            print(f"Threshold optimized with best F1: {best_f1:.6f}; best τ: {best_τ:.6f}")


def _opt_τ_and_f1(y: np.typing.ArrayLike, y_probs: np.ndarray) -> tuple[float, float]:
    y = np.array(y)
    precision, recall, τs = precision_recall_curve(y, y_probs)

    numerator = 2 * precision * recall
    denominator = precision + recall

    f1s = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_idx = np.argmax(f1s)
    best_f1 = f1s[best_idx]
    # Handle edge case where best_idx is the last element (τ is undefined there in sklearn)
    best_τ = τs[best_idx] if best_idx < len(τs) else τs[-1]

    return float(best_τ), float(best_f1)
