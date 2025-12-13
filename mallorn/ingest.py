from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from typer import Typer

from .common import DF_SPLITS, INGESTED_DIR, ORIGIN_DIR, DfType, mkdir, mkpb, mksb


typer = Typer()


@typer.command()
def ingest(dl_dir: Path = ORIGIN_DIR, ingest_dir: Path = INGESTED_DIR):
    with mksb() as sb:
        with sb.job("Calling `download()`"):
            download(dl_dir=dl_dir)

        with sb.job("Calling `reformat(df_type='train')`"):
            reformat(df_type="train", dl_dir=dl_dir, ingest_dir=ingest_dir)

        with sb.job("Calling `reformat(df_type='test')`"):
            reformat(df_type="test", dl_dir=dl_dir, ingest_dir=ingest_dir)


@typer.command()
def download(dl_dir: Path = ORIGIN_DIR):
    """
    Download the dataset from Kaggle.
    """
    with mksb() as sb:
        with sb.job("Instantiating Kaggle API"):
            kaggle = _load_kaggle()

        with sb.job(f"Downloading Kaggle dataset to `{dl_dir.as_posix()}`"):
            kaggle.competition_download_files(
                competition="mallorn-astronomical-classification-challenge",
                path=dl_dir.as_posix(),
            )

        with sb.job("Unzipping downloaded archives"):
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


@typer.command()
def reformat(
    df_type: DfType, dl_dir: Path = ORIGIN_DIR, ingest_dir: Path = INGESTED_DIR
):
    """
    Reformat the downloaded dataset in `indirpath` and save to `outdirpath`.
    """
    with mksb() as sb:
        with sb.job(f"Reformatting `{(dl_dir / f'{df_type}_log.csv').as_posix()}`"):
            meta_df = pd.read_csv(dl_dir / f"{df_type}_log.csv")
            meta_df.set_index("object_id")
            meta_df.rename(
                columns={
                    "Z": "z",
                    "Z_err": "z_err",
                    "EBV": "ebv",
                }
            )

        with sb.job(
            f"Saving reformatted dataset to `{(ingest_dir / f'{df_type}_meta.parquet').as_posix()}`"
        ):
            mkdir(ingest_dir)
            meta_df.to_parquet(ingest_dir / f"{df_type}_meta.parquet")

        with mkpb(total=len(DF_SPLITS) * 2) as pb:
            for df_split in DF_SPLITS:
                with pb.job(
                    f"Reformatting `{(dl_dir / f'{df_split}/{df_type}_full_lightcurves.csv').as_posix()}`"
                ):
                    obs_df = pd.read_csv(
                        dl_dir / f"{df_split}/{df_type}_full_lightcurves.csv"
                    )
                    obs_df.rename(
                        columns={
                            "Time (MJD)": "mjd",
                            "Flux": "flux",
                            "Flux_err": "flux_err",
                            "Filter": "filter",
                        }
                    )

                with pb.job(
                    f"Saving reformatted dataset to `{(ingest_dir / f'{df_split}/{df_type}_obs.parquet').as_posix()}`"
                ):
                    mkdir(ingest_dir / f"{df_split}")
                    obs_df.to_parquet(ingest_dir / f"{df_split}/{df_type}_obs.parquet")
