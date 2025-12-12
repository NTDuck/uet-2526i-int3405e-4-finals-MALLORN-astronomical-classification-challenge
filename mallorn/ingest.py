from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from .common import DF_SPLITS, DL_DIR, INGEST_DIR, DfType, mkdir


def ingest(dl_dir: Path = DL_DIR, ingest_dir: Path = INGEST_DIR):
    download(dl_dir=dl_dir)
    print("Datasets downloaded")

    reformat(df_type="train", dl_dir=dl_dir, ingest_dir=ingest_dir)
    print("`train` datasets reformatted")

    reformat(df_type="test", dl_dir=dl_dir, ingest_dir=ingest_dir)
    print("`test` datasets reformatted")


def download(dl_dir: Path = DL_DIR):
    """
    Download the dataset from Kaggle.
    """
    kaggle = _load_kaggle()
    kaggle.competition_download_files(
        competition="mallorn-astronomical-classification-challenge",
        path=dl_dir.as_posix(),
    )

    _unzip(dl_dir)


def _load_kaggle() -> KaggleApi:
    """
    Load an instance of `KaggleApi`.
    """
    load_dotenv()

    kaggle = KaggleApi(enable_oauth=False)
    kaggle.read_config_environment()
    kaggle.authenticate()

    return kaggle


def _unzip(dir: Path):
    """
    Recursively extract all `.zip` files in `dir`, then remove them.
    """
    for filepath in dir.rglob("*.zip"):
        with ZipFile(filepath) as zip:
            zip.extractall(filepath.parent)
        filepath.unlink()


def reformat(df_type: DfType, dl_dir: Path = DL_DIR, ingest_dir: Path = INGEST_DIR):
    """
    Reformat the downloaded dataset in `indirpath` and save to `outdirpath`.
    """
    log_df = pd.read_csv(dl_dir / f"{df_type}_log.csv")
    log_df.set_index("object_id")
    log_df.rename(
        columns={
            "Z": "z",
            "Z_err": "z_err",
            "EBV": "ebv",
        }
    )

    mkdir(ingest_dir)
    log_df.to_parquet(ingest_dir / f"{df_type}_log.parquet")

    for df_split in DF_SPLITS:
        flc_df = pd.read_csv(dl_dir / f"{df_split}/{df_type}_full_lightcurves.csv")
        flc_df.rename(
            columns={
                "Time (MJD)": "mjd",
                "Flux": "flux",
                "Flux_err": "flux_err",
                "Filter": "filter",
            }
        )

        mkdir(ingest_dir / f"{df_split}")
        flc_df.to_parquet(ingest_dir / f"{df_split}/{df_type}_flc.parquet")
