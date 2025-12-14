from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from typer import Typer

from .common import DF_SPLITS, ING_DIR, ORG_DIR, DfSplit, DfType, mkdir, mkpb, mksb


typer = Typer()


@typer.command()
def ingest(org_dir: Path = ORG_DIR, ing_dir: Path = ING_DIR):
    with mksb() as sb:
        with sb.job("Calling `download()`"):
            download(org_dir=org_dir)

        with sb.job("Calling `reformat(df_type='train')`"):
            format(df_type="train", org_dir=org_dir, ing_dir=ing_dir)

        with sb.job("Calling `reformat(df_type='test')`"):
            format(df_type="test", org_dir=org_dir, ing_dir=ing_dir)


@typer.command()
def download(org_dir: Path = ORG_DIR):
    """
    Download the dataset from Kaggle.
    """
    with mksb() as sb:
        with sb.job("Instantiating Kaggle API"):
            kaggle = _load_kaggle()

        with sb.job(f"Downloading Kaggle dataset to `{org_dir.as_posix()}`"):
            kaggle.competition_download_files(
                competition="mallorn-astronomical-classification-challenge",
                path=org_dir.as_posix(),
            )

        with sb.job("Unzipping downloaded archives"):
            _unzip(org_dir)


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
def format(df_type: DfType, org_dir: Path = ORG_DIR, ing_dir: Path = ING_DIR):
    """
    Reformat the downloaded dataset in `indirpath` and save to `outdirpath`.
    """
    with mksb() as sb:
        with sb.job(f"Reformatting `{(org_dir / f'{df_type}_log.csv').as_posix()}`"):
            meta_df = pd.read_csv(org_dir / f"{df_type}_log.csv")
            meta_df.set_index("object_id", inplace=True)
            meta_df.rename(
                columns={
                    "Z": "z",
                    "Z_err": "z_err",
                    "EBV": "ebv",
                },
                inplace=True,
            )  # format: skip

        with sb.job(f"Saving reformatted dataset to `{(ing_dir / f'{df_type}_meta.parquet').as_posix()}`"):
            meta_df.to_parquet(mkdir(ing_dir) / f"{df_type}_meta.parquet")

        with mkpb(total=len(DF_SPLITS) * 2) as pb:
            for df_split in DF_SPLITS:
                with pb.job(f"Reformatting `{(org_dir / f'{df_split}/{df_type}_full_lightcurves.csv').as_posix()}`"):
                    obs_df = pd.read_csv(org_dir / f"{df_split}/{df_type}_full_lightcurves.csv")
                    obs_df.rename(
                        columns={
                            "Time (MJD)": "mjd",
                            "Flux": "flux",
                            "Flux_err": "flux_err",
                            "Filter": "filter",
                        },
                        inplace=True,
                    )  # format: skip

                with pb.job(f"Saving reformatted dataset to `{(ing_dir / f'{df_split}/{df_type}_obs.parquet').as_posix()}`"):
                    obs_df.to_parquet(mkdir(ing_dir / df_split) / f"{df_type}_obs.parquet")


def load_meta_df(df_type: DfType, ing_dir: Path = ING_DIR) -> pd.DataFrame:
    return pd.read_parquet(ing_dir / f"{df_type}_meta.parquet")


def load_obs_df(df_type: DfType, df_split: DfSplit, ing_dir: Path = ING_DIR) -> pd.DataFrame:
    return pd.read_parquet(ing_dir / f"{df_split}/{df_type}_obs.parquet")
