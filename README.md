# uet-2526i-int3405e-4-finals-MALLORN-astronomical-classification-challenge

> https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/

## Prerequisites

- Python 3.12
- Poetry 2.2.1

## How to install

Running `$ poetry run mallorn featurize-all` requires a Linux environment. Use
WSL on Windows.

```cmd
$ wsl
$ sudo apt-get update
$ sudo apt-get install libgsl-dev pipx

$ pipx install poetry
$ pipx ensurepath
```

```cmd
$ poetry install
$ poetry run pre-commit install
```

## How to run

```cmd
$ poetry run mallorn ingest
$ poetry run mallorn augment-all
$ poetry run mallorn featurize-all
$ poetry run mallorn opt-cb
$ poetry run mallorn opt-τ
$ poetry run mallorn predict
```
