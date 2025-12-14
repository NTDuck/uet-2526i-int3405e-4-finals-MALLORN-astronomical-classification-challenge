from typer import Typer

from . import augment, ingest, featurize


_typer = Typer()

_typer.add_typer(ingest.typer)
_typer.add_typer(augment.typer)
_typer.add_typer(featurize.typer)


def __call__():
    _typer.__call__()
