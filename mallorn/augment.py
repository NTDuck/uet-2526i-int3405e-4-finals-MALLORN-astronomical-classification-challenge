import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from extinction import fitzpatrick99
from typer import Typer

from .common import AUG_DIR, ING_DIR, DfType, mksb
from .ingest import load_meta_df as _load_meta_df
from .ingest import load_obs_dfs as _load_obs_dfs


warnings.filterwarnings("ignore", module="astropy")

typer = Typer()


@typer.command(name="augment-all")
def augment_all(ing_dir: Path = ING_DIR, aug_dir: Path = AUG_DIR):
    with mksb() as sb:
        with sb.job("Calling `augment(df_type='train')`"):
            augment(df_type="train", ing_dir=ing_dir, aug_dir=aug_dir)

        with sb.job("Calling `augment(df_type='test')`"):
            augment(df_type="test", ing_dir=ing_dir, aug_dir=aug_dir)


@typer.command()
def augment(df_type: DfType, ing_dir: Path = ING_DIR, aug_dir: Path = AUG_DIR):
    with mksb() as sb:
        with sb.job("Loading ingested datasets"):
            meta_df = _load_meta_df(df_type=df_type, ing_dir=ing_dir)
            obs_df = _load_obs_dfs(df_type=df_type, ing_dir=ing_dir)

        with sb.job("Calculating Distance Modulus `μ`"):
            z = meta_df["z"].to_numpy(dtype=np.float64)
            z = np.clip(z, np.finfo(np.float64).eps, np.finfo(np.float64).max)

            meta_df["μ"] = Planck18.distmod(z).value  # pyright: ignore[reportAttributeAccessIssue]

        with sb.job("Applying Flux De-extinction"):
            eff_wl = obs_df["filter"].map(_EFF_WLS).to_numpy(dtype=np.float64)
            ebv = obs_df["object_id"].map(meta_df["ebv"]).to_numpy(dtype=np.float64)

            a_λ = fitzpatrick99(eff_wl, a_v=1.0, r_v=3.1)
            factor = np.pow(10, 0.4 * a_λ * ebv)

            obs_df["flux"] *= factor
            obs_df["flux_err"] *= factor

        with sb.job("Handling Negative Fluxes with Luptitudes"):
            # https://classic.sdss.org/dr7/algorithms/photometry.php
            obs_df["flux_lupt"] = -2.5 / np.log(10) * np.arcsinh(obs_df["flux"] / (2 * obs_df["flux_err"]))

        if df_type == "train":
            with sb.job("Applying Gaussian Noise to Redshifts"):
                z_err_test = _load_meta_df(df_type="test", ing_dir=ing_dir, columns=["z_err"])["z_err"]
                σ_z = z_err_test.std()

                noise = np.random.normal(loc=0.0, scale=σ_z, size=len(meta_df))
                meta_df["z"] += noise

            with sb.job("Re-calculating Distance Modulus `μ`"):
                z = meta_df["z"].to_numpy(dtype=np.float64)
                z = np.clip(z, np.finfo(np.float64).eps, np.finfo(np.float64).max)

                meta_df["μ"] = Planck18.distmod(z).value  # pyright: ignore[reportAttributeAccessIssue]

        with sb.job("Saving augmented datasets"):
            meta_df.to_parquet(aug_dir / f"{df_type}_meta_aug.parquet")
            obs_df.to_parquet(aug_dir / f"{df_type}_obs_aug.parquet")


_EFF_WLS = {
    "u": 3641.0,
    "g": 4704.0,
    "r": 6155.0,
    "i": 7504.0,
    "z": 8695.0,
    "y": 10056.0,
}


def load_meta_aug_df(df_type: DfType, aug_dir: Path = AUG_DIR, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(aug_dir / f"{df_type}_meta_aug.parquet", **kwargs)


def load_obs_aug_df(df_type: DfType, aug_dir: Path = AUG_DIR) -> pd.DataFrame:
    return pd.read_parquet(aug_dir / f"{df_type}_obs_aug.parquet")
