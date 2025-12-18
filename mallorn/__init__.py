from typer import Typer

from . import augment, featurize, ingest, optimize, predict


_typer = Typer()

_typer.add_typer(ingest.typer)
_typer.add_typer(augment.typer)
_typer.add_typer(featurize.typer)
_typer.add_typer(optimize.typer)
_typer.add_typer(predict.typer)


def __call__():
    _typer.__call__()
