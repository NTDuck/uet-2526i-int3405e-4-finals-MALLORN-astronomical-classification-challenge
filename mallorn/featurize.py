from pathlib import Path
import numpy as np
import pandas as pd
from typer import Typer

from .augment import load_meta_aug_df as _load_meta_aug_df, load_obs_aug_df as _load_obs_aug_df
from .common import AUG_DIR, FEATS_DIR, DfType, mkdir


typer = Typer()


# @typer.command()
# def featurize(df_type: DfType, aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
#     meta_df = _load_meta_aug_df(df_type, aug_dir)
#     obs_df = _load_obs_aug_df(df_type, aug_dir)

#     mrg_df = obs_df.merge(meta_df, left_on="object_id", right_index=True)
#     feats_df = meta_df.drop(columns=["SpecType", "English Translation", "split"])

#     # Parametric Villar/Bazin Fits using Likelihood Ratio
#     bazin_feats_df = mrg_df.groupby(["object_id", "filter"]).apply(_featurize_bazin).unstack("filter")
#     bazin_feats_df.columns = [f"{feat}_{filt}" for feat, filt in bazin_feats_df.columns]
#     feats_df.update(bazin_feats_df)

#     feats_df.to_parquet(mkdir(feats_dir) / f"{df_type}_feats.parquet")

# def _featurize_bazin(mrgs: pd.DataFrame) -> pd.Series:
#     pass

# def _bazin(A, B, t_0, )

@typer.command()
def foo():
    from feets import FeatureSpace
    fs = FeatureSpace()
    print(f"Available extractors: {len(fs.get_available_features())}")


def _tde(t, t_0, t_peak, f_peak, α):
    assert(t_peak <= t_0)
    return f_peak * np.pow((t - t_0) / (t_peak - t_0), -α)