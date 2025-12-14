from pathlib import Path

from .augment import load_meta_aug_df, load_obs_aug_df
from .common import AUG_DIR, FEATS_DIR, DfType


def featurize(df_type: DfType, aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
    meta_df = load_meta_aug_df(df_type, aug_dir)
    obs_df = load_obs_aug_df(df_type, aug_dir)


# Things to do here
# Extinction correction using mwebv + configurable R_V
# Luptitudes
# Augmentations: redshift-time, flux dimming by distance, GP sampling for rare TDEs
