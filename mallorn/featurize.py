from pathlib import Path

import feets
import numpy as np
import pandas as pd
from typer import Typer

from .augment import load_meta_aug_df as _load_meta_aug_df
from .augment import load_obs_aug_df as _load_obs_aug_df
from .common import AUG_DIR, FEATS_DIR, DfType, mkdir, mksb


typer = Typer()


@typer.command()
def featurize(df_type: DfType, aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
    with mksb() as sb:
        with sb.job("Loading augmented datasets"):
            meta_df = _load_meta_aug_df(df_type, aug_dir)
            obs_df = _load_obs_aug_df(df_type, aug_dir)

            feats_df = meta_df.drop(columns=["SpecType", "English Translation", "split"])

        with sb.job("Merging augmented datasets"):
            mrg_df = obs_df.merge(meta_df, left_on="object_id", right_index=True)
            mrg_df.sort_values(["object_id", "filter", "mjd"], inplace=True)

        with sb.job("Extracting Bazin features"):
            bazin_feats_df = mrg_df.groupby(["object_id", "filter"])[["mjd", "flux", "flux_err"]].apply(_bazin).unstack("filter")
            bazin_feats_df.columns = [f"{feat}_{filt}" for feat, filt in bazin_feats_df.columns]

            feats_df = feats_df.join(bazin_feats_df)

        print(feats_df)

        with sb.job("Saving featurized dataset"):
            feats_df.to_parquet(mkdir(feats_dir) / f"{df_type}_feats.parquet")


def _bazin(mrgs: pd.DataFrame):
    mrgs.dropna(subset=["flux", "flux_err"], inplace=True)

    try:
        feats = feets.extractors.BazinFit().extract(mrgs["mjd"].to_numpy(dtype=np.float64), mrgs["flux"].to_numpy(dtype=np.float64), mrgs["flux_err"].to_numpy(dtype=np.float64))
        return pd.Series(feats)
    except:
        return pd.Series([np.nan] * len(_BAZIN_FEATS), index=_BAZIN_FEATS)


_BAZIN_FEATS = ["BazinFit_Amplitude", "BazinFit_Baseline", "BazinFit_ReferenceTime", "BazinFit_RiseTime", "BazinFit_FallTime", "BazinFit_ReducedChi2"]

# @typer.command()
# def foo():
#     fs = feets.FeatureSpace(data={"magnitude", "time"})
#     print(f"Selected features: {list(fs.selected_features)}")


def _tde(t, t_0, t_peak, f_peak, α):
    assert t_peak <= t_0
    return f_peak * np.pow((t - t_0) / (t_peak - t_0), -α)
