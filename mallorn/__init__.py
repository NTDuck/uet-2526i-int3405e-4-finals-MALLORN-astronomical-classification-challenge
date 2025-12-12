from typer import Typer


typer = Typer()


@typer.command()
def hello(name: str):
    print(f"Hello {name}")


@typer.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")


def run():
    typer()
