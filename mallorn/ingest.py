from pathlib import Path
from zipfile import ZipFile

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


def download(dirpath: str = "./resources/kaggle"):
    """
    Download the dataset from Kaggle.
    """
    kaggle = _load_kaggle()
    kaggle.competition_download_files(
        competition="mallorn-astronomical-classification-challenge", path=dirpath
    )

    _unzip(dirpath)


def _load_kaggle() -> KaggleApi:
    """
    Load an instance of `KaggleApi`.
    """
    load_dotenv()

    kaggle = KaggleApi(enable_oauth=False)
    kaggle.read_config_environment()
    kaggle.authenticate()

    return kaggle


def _unzip(dirpath: str):
    """
    Recursively extract all `.zip` files in `dirpath`, then remove them.
    """
    for filepath in Path(dirpath).rglob("*.zip"):
        with ZipFile(filepath) as zip:
            zip.extractall(filepath.parent)
        filepath.unlink()
