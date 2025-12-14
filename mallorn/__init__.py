from typer import Typer

from . import augment, ingest


_typer = Typer()

_typer.add_typer(ingest.typer)
_typer.add_typer(augment.typer)


def __call__():
    _typer.__call__()
