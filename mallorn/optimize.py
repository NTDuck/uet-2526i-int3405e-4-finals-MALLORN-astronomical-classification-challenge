import json
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typer import Typer

from .common import FEATS_DIR, MODELS_DIR, SEED, mkdir, mksb, now
from .featurize import load_feats_df as _load_feats_df


typer = Typer()


@typer.command(name="opt-cb")
def opt_cb(n_trials: int = 300, feats_dir: Path = FEATS_DIR, models_dir: Path = MODELS_DIR):
    with mksb() as sb:
        with sb.job("Loading featurized dataset"):
            feats_df = _load_feats_df(df_type="train", feats_dir=feats_dir)
            X = feats_df.drop(columns=["target"])
            y = feats_df["target"]

            # Calculate imbalance ratio for scale_pos_weight
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    def objective(trial: optuna.Trial):
        # Hyperparameter search space
        params = {
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            # Strategy: Optimize Logloss with weighting, measure AUC
            # "task_type": "GPU",
            "loss_function": "Logloss",
            "scale_pos_weight": scale_pos,
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": SEED,
            "allow_writing_files": False,
        }

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        auc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

            pred_probs = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, pred_probs))

        return np.mean(auc_scores)

    study = optuna.create_study(storage="sqlite:///optuna.db", direction="maximize", study_name=f"cb_study_{now()}")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)  # pyright: ignore[reportArgumentType]

    best_auc = study.best_value
    best_params = study.best_params

    # Add fixed params back
    best_params.update(
        {
            # "task_type": "GPU",
            "loss_function": "Logloss",
            "scale_pos_weight": scale_pos,
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": SEED,
            "allow_writing_files": False,
        }
    )

    print(f"Catboost optimized with best AUC: {best_auc:.6f}")

    with mksb() as sb:
        with sb.job("Saving optimized hyperparameters"):
            with open(mkdir(models_dir / "catboost") / f"params-{now()}.json", "w") as file:
                json.dump(best_params, file, indent=4)


@typer.command(name="opt-τ")
def opt_τ(feats_dir: Path = FEATS_DIR, models_dir: Path = MODELS_DIR):
    with mksb() as sb:
        with sb.job("Loading featurized dataset"):
            feats_df = _load_feats_df(df_type="train", feats_dir=feats_dir)
            X = feats_df.drop(columns=["target"])
            y = feats_df["target"]

            # Load latest params
            param_files = list((models_dir / "catboost").glob("params-*.json"))
            if not param_files:
                raise FileNotFoundError("No params file found. Run opt-cb first.")

            with open(max(param_files)) as file:
                cb_params = json.load(file)

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
                json.dump(
                    {
                        "best_τ": best_τ,
                        "best_f1": best_f1,
                    },
                    file,
                    indent=4,
                )

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
