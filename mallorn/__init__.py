from pathlib import Path

from typer import Typer

from .common import DL_DIR, INGEST_DIR, DfType
from .ingest import download, ingest, reformat


_typer = Typer()


@_typer.command(name="ingest")
def _ingest(dl_dir: Path = DL_DIR, ingest_dir: Path = INGEST_DIR):
    ingest(dl_dir, ingest_dir)  # pyright: ignore[reportCallIssue]


@_typer.command()
def _download(dl_dir: Path = DL_DIR):
    download(dl_dir)


@_typer.command(name="reformat")
def _reformat(df_type: DfType, dl_dir: Path = DL_DIR, ingest_dir: Path = INGEST_DIR):
    reformat(df_type, dl_dir, ingest_dir)


def run():
    _typer()
