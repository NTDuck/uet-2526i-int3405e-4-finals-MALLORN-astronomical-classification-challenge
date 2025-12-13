from typer import Typer

from . import ingest


_typer = Typer()

_typer.add_typer(ingest.typer)


def __call__():
    _typer.__call__()
