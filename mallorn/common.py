from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


def mkdir(dir: Path) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%z")


@contextmanager
def mksb():
    sb = _SpinnerBar()
    with sb._progress:
        yield sb


class _SpinnerBar:
    def __init__(self):
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            transient=True,
        )

    @contextmanager
    def job(self, description: str):
        task_id = self._progress.add_task(description, total=None)
        try:
            yield task_id
        finally:
            self._progress.update(task_id, visible=False)


@contextmanager
def mkpb(total: float):
    pb = _ProgressBar(total=total)
    with pb._progress:
        yield pb


class _ProgressBar:
    def __init__(self, total: float):
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        self._task_id = self._progress.add_task("", total=total)

    @contextmanager
    def job(self, description: str):
        self._progress.update(self._task_id, description=description)
        try:
            yield
        finally:
            self._progress.update(self._task_id, advance=1)


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


ORG_DIR = Path("./artifacts/origin")
ING_DIR = Path("./artifacts/interim")
AUG_DIR = Path("./artifacts/augmented")
FEATS_DIR = Path("./artifacts/features")
MODELS_DIR = Path("./artifacts/models")
PREDS_DIR = Path("./artifacts/predictions")

SEED = 67
