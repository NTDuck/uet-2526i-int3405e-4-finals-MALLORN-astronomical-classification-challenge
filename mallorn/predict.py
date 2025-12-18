import json
import re
from collections import defaultdict
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from typer import Typer

from .common import FEATS_DIR, MODELS_DIR, PREDS_DIR, mkdir, mksb, now
from .featurize import load_feats_df as _load_feats_df


typer = Typer()


@typer.command()
def predict(feats_dir: Path = FEATS_DIR, models_dir: Path = MODELS_DIR, preds_dir: Path = PREDS_DIR):
    with mksb() as sb:
        with sb.job("Loading featurized datasets"):
            test_feats_df = _load_feats_df(df_type="test", feats_dir=feats_dir)

            with open(max((models_dir / "τ").glob("*.json"))) as file:
                τ = json.load(file)["best_τ"]

        with sb.job("Ensembling Catboost predictions"):
            cb_models = _load_cb_models(models_dir=models_dir)
            avg_probs = np.mean([model.predict_proba(test_feats_df)[:, 1] for model in cb_models], axis=0)

        with sb.job(f"Applying τ={τ:6f}"):
            y_pred = (avg_probs > τ).astype(int)

        with sb.job(f"Saving predictions to {preds_dir / f'predictions-{now()}.csv'}"):
            preds_df = pd.DataFrame({"object_id": test_feats_df.index, "target": y_pred})
            preds_df.to_csv(mkdir(preds_dir) / f"predictions-{now()}.csv", index=False)


def _load_cb_models(models_dir: Path) -> list[cb.CatBoostClassifier]:
    fold_re = re.compile(r"model-fold-(\d+)-")

    groups: dict[int, list[Path]] = defaultdict(list)
    for file in (models_dir / "catboost").glob("model-fold-*.cbm"):
        fold = int(fold_re.search(file.name).group(1))  # pyright: ignore[reportOptionalMemberAccess]
        groups[fold].append(file)

    files = (max(files) for files in groups.values())

    return [cb.CatBoostClassifier().load_model(file) for file in files]
