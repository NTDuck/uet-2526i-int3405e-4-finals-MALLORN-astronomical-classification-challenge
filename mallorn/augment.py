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

        # 1. De-extinction (Standard)
        with sb.job("Applying Flux De-extinction"):
            eff_wl = obs_df["filter"].map(_EFF_WLS).to_numpy(dtype=np.float64)
            ebv = obs_df["object_id"].map(meta_df["ebv"]).to_numpy(dtype=np.float64)
            a_lambda = fitzpatrick99(eff_wl, a_v=1.0, r_v=3.1)
            factor = np.power(10, 0.4 * a_lambda * ebv)
            obs_df["flux"] *= factor
            obs_df["flux_err"] *= factor

        # 2. Physics-Informed Augmentation (Train Only - DYNAMIC BALANCING)
        if df_type == "train":
            with sb.job("Augmenting TDEs (Dynamic Balancing Strategy)"):
                tde_ids = meta_df[meta_df["target"] == 1].index
                non_tde_count = (meta_df["target"] == 0).sum()

                # Friend's Strategy: Balance TDEs to be ~33% of Non-TDE count (1:3 ratio)
                target_tde_count = int(non_tde_count // 3)

                if len(tde_ids) > 0:
                    current_tde_count = len(tde_ids)
                    # How many clones per object?
                    n_aug_per_obj = max(1, (target_tde_count - current_tde_count) // current_tde_count)

                    print(f"   [Info] Original TDEs: {current_tde_count}. Target: {target_tde_count}.")
                    print(f"   [Info] Augmenting each TDE {n_aug_per_obj} times.")

                    z_vals = np.clip(meta_df.loc[tde_ids, "z"].values, 0.001, None)  # pyright: ignore
                    d_lum_orig = Planck18.luminosity_distance(z_vals).value  # pyright: ignore
                    dist_map = dict(zip(tde_ids, d_lum_orig))

                    obs_groups = obs_df[obs_df["object_id"].isin(tde_ids)].groupby("object_id")

                    new_meta, new_obs = [], []

                    for obj_id in tde_ids:
                        if obj_id not in obs_groups.groups:
                            continue
                        orig_meta = meta_df.loc[obj_id]
                        orig_obs = obs_groups.get_group(obj_id)
                        z_old = max(0.001, orig_meta["z"])
                        d_old = dist_map[obj_id]

                        # Generate new redshifts (TDEs usually 0.01 - 0.5)
                        new_zs = np.random.uniform(0.01, 0.5, n_aug_per_obj)

                        for i, z_new in enumerate(new_zs):
                            new_id = f"{obj_id}_aug_{i}"

                            # Physics: Time Dilation & Flux Scaling
                            t_scale = (1 + z_new) / (1 + z_old)

                            d_new = Planck18.luminosity_distance(z_new).value  # pyright: ignore
                            f_scale = (d_old / d_new) ** 2

                            # Meta
                            m = orig_meta.copy()
                            m.name = new_id
                            m["z"] = z_new
                            new_meta.append(m)

                            # Obs
                            o = orig_obs.copy()
                            o["object_id"] = new_id
                            o["mjd"] = o["mjd"] * t_scale
                            o["flux"] = o["flux"] * f_scale
                            o["flux_err"] = o["flux_err"] * f_scale
                            new_obs.append(o)

                    if new_meta:
                        meta_df = pd.concat([meta_df, pd.DataFrame(new_meta)])
                        obs_df = pd.concat([obs_df, pd.concat(new_obs)])

            # Noise Injection for robustness
            with sb.job("Adding Redshift Noise"):
                sigma_z = 0.01
                mask = ~meta_df.index.astype(str).str.contains("_aug_")
                noise = np.random.normal(0, sigma_z, mask.sum())
                meta_df.loc[mask, "z"] += noise

        # 3. Derived Columns
        with sb.job("Calculating Derived Columns"):
            # Luptitude (arcsinh magnitude) is robust against negative flux
            obs_df["flux_lupt"] = -2.5 / np.log(10) * np.arcsinh(obs_df["flux"] / (2 * obs_df["flux_err"] + 1e-9))

            # Distance Modulus
            z = np.clip(meta_df["z"].values, 1e-5, None)  # pyright: ignore
            meta_df["μ"] = Planck18.distmod(z).value  # pyright: ignore

        with sb.job("Saving Parquet"):
            meta_df.index.name = "object_id"
            meta_df.to_parquet(aug_dir / f"{df_type}_meta_aug.parquet")
            obs_df.to_parquet(aug_dir / f"{df_type}_obs_aug.parquet")


_EFF_WLS = {"u": 3641.0, "g": 4704.0, "r": 6155.0, "i": 7504.0, "z": 8695.0, "y": 10056.0}


def load_meta_aug_df(df_type: DfType, aug_dir: Path = AUG_DIR, **kwargs):
    return pd.read_parquet(aug_dir / f"{df_type}_meta_aug.parquet", **kwargs)


def load_obs_aug_df(df_type: DfType, aug_dir: Path = AUG_DIR):
    return pd.read_parquet(aug_dir / f"{df_type}_obs_aug.parquet")
