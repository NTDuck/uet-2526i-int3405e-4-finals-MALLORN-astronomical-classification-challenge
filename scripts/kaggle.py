from pathlib import Path
from zipfile import ZipFile
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


def download(dirpath: str = "./resources/kaggle"):
    """
    Download the dataset from Kaggle.
    """
    kaggle = __load_kaggle()
    kaggle.competition_download_files(competition="mallorn-astronomical-classification-challenge", path=dirpath)

    __unzip(dirpath)


def submit():
    """
    Submit the dataset to Kaggle.
    """
    pass


def __load_kaggle() -> KaggleApi:
    """
    Load an instance of `KaggleApi`.
    """
    load_dotenv()
    
    kaggle = KaggleApi()
    kaggle.read_config_environment()
    kaggle.authenticate()

    return kaggle


def __unzip(dirpath: str):
    """
    Recursively extract all `.zip` files in `dirpath`, then remove them.
    """
    for filepath in Path(dirpath).rglob("*.zip"):
        with ZipFile(filepath) as zip:
            zip.extractall(filepath.parent)
        filepath.unlink()
