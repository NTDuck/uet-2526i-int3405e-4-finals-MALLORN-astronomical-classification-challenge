from pathlib import Path

import numpy as np
import pandas as pd
from extinction import fitzpatrick99
from typer import Typer

from .common import AUG_DIR, ING_DIR, DfType, mksb
from .ingest import load_meta_df as _load_meta_df
from .ingest import load_obs_dfs as _load_obs_dfs


typer = Typer()


@typer.command()
def augment(df_type: DfType, ing_dir: Path = ING_DIR, aug_dir: Path = AUG_DIR):
    with mksb() as sb:
        with sb.job("Loading ingested datasets"):
            meta_df = _load_meta_df(df_type, ing_dir)
            obs_df = pd.concat(_load_obs_dfs(df_type, ing_dir).values())  # fmt: skip

        with sb.job("Handling Cosmic Time Dialation"):
            # https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2022.826188/full
            obs_df["mjd"] /= 1 + obs_df["object_id"].map(meta_df["z"])

        with sb.job("Applying Flux De-extinction"):
            A_λ = fitzpatrick99(obs_df["filter"].map(_EFF_WLS).to_numpy(dtype=np.float64), a_v=1, r_v=3.1)
            obs_df["flux"] *= np.pow(10, 0.4 * A_λ * obs_df["object_id"].map(meta_df["ebv"]))

        with sb.job("Handling Negative Fluxes via Luptitudes"):
            obs_df["flux_lupt"] = -2.5 / np.log(10) * np.arcsinh(obs_df["flux"] / (2 * obs_df["flux_err"]))

        if df_type == "train":
            with sb.job("Applying Gaussian Noise to Redshifts"):
                σ_z_err = _load_meta_df(df_type="test", ing_dir=ing_dir, columns=["z_err"])["z_err"].std()
                noise = np.random.normal(loc=0.0, scale=σ_z_err, size=len(meta_df))
                meta_df["z"] = (meta_df["z"] + noise).clip(np.finfo(float).eps)

        # TODO Resample
        # https://www.sciencedirect.com/science/article/abs/pii/S0031320312001471
        # https://towardsdatascience.com/imbalanced-data-stop-using-roc-auc-and-use-auprc-instead-46af4910a494/

        with sb.job("Saving augmented datasets"):
            meta_df.to_parquet(aug_dir / f"{df_type}_meta_aug.parquet")
            obs_df.to_parquet(aug_dir / f"{df_type}_obs_aug.parquet")


# @deprecated
# def _iter(meta_df: pd.DataFrame, obs_df: pd.DataFrame) -> Iterator[Tuple[Hashable, pd.Series, pd.Series]]:
#     obs_grps = obs_df.groupby("object_id")

#     for obj_id, meta in meta_df.iterrows():
#         for _, obs_grp in obs_grps.get_group(obj_id).groupby("filter"):
#             for _, obs in obs_grp.iterrows():
#                 yield obj_id, meta, obs


_EFF_WLS = {
    "u": np.array([3641]),
    "g": np.array([4704]),
    "r": np.array([6155]),
    "i": np.array([7504]),
    "z": np.array([8695]),
    "y": np.array([10056]),
}


def load_meta_aug_df(df_type: DfType, aug_dir: Path = AUG_DIR, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(aug_dir / f"{df_type}_meta_aug.parquet", **kwargs)


def load_obs_aug_df(df_type: DfType, aug_dir: Path = AUG_DIR) -> pd.DataFrame:
    return pd.read_parquet(aug_dir / f"{df_type}_obs_aug.parquet")
