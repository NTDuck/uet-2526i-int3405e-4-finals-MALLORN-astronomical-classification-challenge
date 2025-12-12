from typer import Typer

from .ingest import download as ingest_download

typer = Typer()


@typer.command(name="ingest-download")
def download():
    ingest_download()


def run():
    typer()
