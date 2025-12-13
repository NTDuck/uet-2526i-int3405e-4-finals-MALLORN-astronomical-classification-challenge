from pathlib import Path

import pandas as pd

import avocado

from .common import AUGMENTED_DIR, DF_SPLITS, INGESTED_DIR


def augment(ingest_dir: Path = INGESTED_DIR, aug_dir: Path = AUGMENTED_DIR):
    train_meta_df = pd.read_parquet(ingest_dir / "train_meta.parquet")
    test_meta_df = pd.read_parquet(ingest_dir / "test_meta.parquet")

    train_flc_df = pd.concat(
        [
            pd.read_parquet(ingest_dir / f"{df_split}/train_flc.parquet")
            for df_split in DF_SPLITS
        ]
    )

    # https://avocado-classifier.readthedocs.io/en/latest/api/avocado.AstronomicalObject.html#avocado.AstronomicalObject
    train_meta_df.rename(
        columns={
            "object_id": "object_id",
            "z": "redshift",
            "ebv": "galactic_mwebv",
        }
    )
    train_flc_df.rename(
        columns={
            "time": "time",
            "filter": "band",
            "flux": "flux",
            "flux_err": "flux_error",
        }
    )

    # TODO Try chunks?
    df = avocado.Dataset("train_avocado", train_meta_df, train_flc_df)
