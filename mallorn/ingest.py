from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from typer import Typer

from .common import ING_DIR, ORG_DIR, DfSplit, DfType, mkdir, mkpb, mksb


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
    load_dotenv()

    kaggle = KaggleApi(enable_oauth=False)
    kaggle.read_config_environment()
    kaggle.authenticate()

    return kaggle


def _unzip(dir: Path):
    for file in dir.rglob("*.zip"):
        with ZipFile(file) as zip:
            zip.extractall(file.parent)
        file.unlink()


@typer.command()
def format(df_type: DfType, org_dir: Path = ORG_DIR, ing_dir: Path = ING_DIR):
    with mksb() as sb:
        with sb.job(f"Reformatting `{(org_dir / f'{df_type}_log.csv').as_posix()}`"):
            meta_df = pd.read_csv(org_dir / f"{df_type}_log.csv")
            meta_df.set_index("object_id", inplace=True)
            meta_df.rename(columns={
                "Z": "z",
                "Z_err": "z_err",
                "EBV": "ebv",
            }, inplace=True)  # fmt: skip

        with sb.job(f"Saving reformatted dataset to `{(ing_dir / f'{df_type}_meta.parquet').as_posix()}`"):
            meta_df.to_parquet(mkdir(ing_dir) / f"{df_type}_meta.parquet")

        df_splits = sorted(meta_df["split"].unique())
        with mkpb(total=len(df_splits) * 2) as pb:
            for df_split in df_splits:
                with pb.job(f"Reformatting `{(org_dir / f'{df_split}/{df_type}_full_lightcurves.csv').as_posix()}`"):
                    obs_df = pd.read_csv(org_dir / f"{df_split}/{df_type}_full_lightcurves.csv")
                    obs_df.rename(columns={
                        "Time (MJD)": "mjd",
                        "Flux": "flux",
                        "Flux_err": "flux_err",
                        "Filter": "filter",
                    }, inplace=True)  # fmt: skip

                with pb.job(f"Saving reformatted dataset to `{(ing_dir / f'{df_split}/{df_type}_obs.parquet').as_posix()}`"):
                    obs_df.to_parquet(mkdir(ing_dir / df_split) / f"{df_type}_obs.parquet")


def load_meta_df(df_type: DfType, ing_dir: Path = ING_DIR, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(ing_dir / f"{df_type}_meta.parquet", **kwargs)


def load_obs_df(df_type: DfType, df_split: DfSplit, ing_dir: Path = ING_DIR, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(ing_dir / f"{df_split}/{df_type}_obs.parquet", **kwargs)


def load_obs_dfs(df_type: DfType, ing_dir: Path = ING_DIR, **kwargs) -> dict[DfSplit, pd.DataFrame]:
    obs_dfs = {}

    for file in ing_dir.glob(f"*/{df_type}_obs.parquet"):
        df_split = file.parent.name
        obs_dfs[df_split] = load_obs_df(df_type, df_split, ing_dir, **kwargs)  # pyright: ignore[reportArgumentType]

    return obs_dfs
