from pathlib import Path
from typing import Literal


def mkdir(dirpath: str | Path) -> Path:
    dirpath_ = Path(dirpath)
    dirpath_.mkdir(parents=True, exist_ok=True)

    return dirpath_


DL_DIR = mkdir("./artifacts/origin")
INGEST_DIR = mkdir("./artifacts/interim")
FEATS_DIR = mkdir("./artifacts/feats")

DF_SPLITS = [
    "split_01",
    "split_02",
    "split_03",
    "split_04",
    "split_05",
    "split_06",
    "split_07",
    "split_08",
    "split_09",
    "split_10",
    "split_11",
    "split_12",
    "split_13",
    "split_14",
    "split_15",
    "split_16",
    "split_17",
    "split_18",
    "split_19",
    "split_20",
]


DfType = Literal["train", "test"]
DfSplit = Literal[
    "split_01",
    "split_02",
    "split_03",
    "split_04",
    "split_05",
    "split_06",
    "split_07",
    "split_08",
    "split_09",
    "split_10",
    "split_11",
    "split_12",
    "split_13",
    "split_14",
    "split_15",
    "split_16",
    "split_17",
    "split_18",
    "split_19",
    "split_20",
]
