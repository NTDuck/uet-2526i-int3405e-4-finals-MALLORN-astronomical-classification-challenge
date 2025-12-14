from pathlib import Path

import pandas as pd
from typer import Typer

from .common import AUG_DIR, DF_SPLITS, ING_DIR, DfType
from .ingest import load_meta_df, load_obs_df


typer = Typer()


@typer.command()
def augment(df_type: DfType, ing_dir: Path = ING_DIR, aug_dir: Path = AUG_DIR):
    meta_df = load_meta_df(df_type, ing_dir)
    obs_df = pd.concat([load_obs_df(df_type, df_split, ing_dir) for df_split in DF_SPLITS])  # fmt: skip

    # TODO Resample
    # https://www.sciencedirect.com/science/article/abs/pii/S0031320312001471
    # https://towardsdatascience.com/imbalanced-data-stop-using-roc-auc-and-use-auprc-instead-46af4910a494/

    meta_df.to_parquet(aug_dir / f"{df_type}_log_aug.parquet")


def load_aug_meta_df(df_type: DfType, aug_dir: Path = AUG_DIR) -> pd.DataFrame:
    return pd.read_parquet(aug_dir / f"{df_type}_log_aug.parquet")
