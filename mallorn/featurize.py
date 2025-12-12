import pandas as pd
from tqdm import tqdm
from pathlib import Path

from .common import DfType, FEATS_DIR, INGEST_DIR, DF_SPLITS

def featurize(df_type: DfType, ingest_dir: Path = INGEST_DIR, feats_dir: Path = FEATS_DIR):
    log_df = pd.read_parquet(ingest_dir / f"{df_type}_log.parquet")

    with tqdm(total=len(log_df), unit="obj") as pb:
        for df_split in DF_SPLITS:


def build_foo(log_row: pd.Series, flc_df: pd.DataFrame) -> dict:
    feats = {}

    feat[{feat_name}] = feat_val

    return feats

where log_row represents 1 row of object, formatted
object_id,z,z_err,ebv,SpecType,English Translation,split,target
Dornhoth_fervain_onodrim,3.049,,0.11,AGN,Trawn Folk (Dwarfs) + northern + Ents (people) ,split_01,0
(please dont use SpecType, English Translation, split)
and flc_df represents n rows of object, formatted
object_id,mjd,flux,flux_err,filter
Dornhoth_fervain_onodrim,63314.4662,-1.63015865,0.36577725,z
Dornhoth_fervain_onodrim,63780.9674,10.49938934,0.25386745,r
Dornhoth_fervain_onodrim,63789.7693,5.86625017,1.55924117,y
(mjd is time)