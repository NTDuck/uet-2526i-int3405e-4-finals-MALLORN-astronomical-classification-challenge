import json
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from typer import Typer

from .common import FEATS_DIR, MODELS_DIR, SEED, mkdir, mksb, now
from .featurize import load_feats_df as _load_feats_df


typer = Typer()


@typer.command(name="opt-cb")
def opt_cb(n_trials: int = 40, feats_dir: Path = FEATS_DIR, models_dir: Path = MODELS_DIR):
    with mksb() as sb:
        with sb.job("Loading featurized dataset"):
            feats_df = _load_feats_df(df_type="train", feats_dir=feats_dir)

            X = feats_df.drop(columns=["target"])
            y = feats_df["target"]

    def objective(trial: optuna.Trial):
        α = trial.suggest_float("α", 0.1, 0.9)
        γ = trial.suggest_float("γ", 0.5, 3.0)

        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "loss_function": f"Focal:focal_alpha={α};focal_gamma={γ}",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": SEED,
            "allow_writing_files": False,
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1s = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))

            pred_probs = model.predict_proba(X_val)[:, 1]
            _, best_f1 = _opt_τ_and_f1(y_val, pred_probs)

            f1s.append(best_f1)

        return np.mean(f1s)

    study = optuna.create_study(storage="sqlite:///optuna.db", direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)  # pyright: ignore[reportArgumentType]

    best_f1 = study.best_value
    best_params = study.best_params

    α = best_params.pop("α")
    γ = best_params.pop("γ")
    best_params.update(
        {
            "loss_function": f"Focal:focal_alpha={α};focal_gamma={γ}",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": SEED,
            "allow_writing_files": False,
        }
    )

    print(f"Catboost optimized with best F1: {best_f1:.6f}")

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

            with open(max((models_dir / "catboost").glob("params-*.json"))) as file:
                cb_params = json.load(file)

        with sb.job("Populating OOF predictions"):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            oof_preds = np.zeros(len(X))

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                cb_model = cb.CatBoostClassifier(**cb_params)
                cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
                cb_model.save_model(mkdir(models_dir / "catboost") / f"model-fold-{fold}-{now()}.cbm")

                oof_preds[val_idx] = cb_model.predict_proba(X_val)[:, 1]

        with sb.job("Optimizing τ"):
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

    # FIX: Use 'where' to prevent division by zero
    numerator = 2 * precision * recall
    denominator = precision + recall

    f1s = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,  # Only divide where denominator is NOT zero
    )

    best_f1_idx = np.argmax(f1s)
    best_f1 = f1s[best_f1_idx]
    best_τ = τs[best_f1_idx] if best_f1_idx < len(τs) else τs[-1]

    return best_τ, best_f1
