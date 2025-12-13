# import pandas as pd
# from tqdm import tqdm
# from pathlib import Path

# from .common import DfType, FEATS_DIR, INGEST_DIR, DF_SPLITS

# def featurize(df_type: DfType, ingest_dir: Path = INGEST_DIR, feats_dir: Path = FEATS_DIR):
#     log_df = pd.read_parquet(ingest_dir / f"{df_type}_log.parquet")

#     with tqdm(total=len(log_df), unit="obj") as pb:
#         for df_split in DF_SPLITS:
