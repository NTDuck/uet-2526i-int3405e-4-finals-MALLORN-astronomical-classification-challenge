import pandas as pd

from typing import Literal
from autogluon.tabular import TabularDataset, TabularPredictor
from pathlib import Path


# THIS IS PURE HALLUCINATION BTW
def main():
    train_df = build_df(type="train")
    test_df = build_df(type="test")

    # predictor = TabularPredictor(label="target").fit(train_df, presets="extreme")

    # Path("./output").mkdir(parents=True, exist_ok=True)

    # pd.DataFrame({
    #     "object_id": test_df["object_id"],
    #     "prediction": predictor.predict(test_df)
    # }).to_csv("./output/submission.csv", index=False)


def build_df(type: Literal["train", "test"]) -> pd.DataFrame:
    log_df = pd.read_csv(f"./resources/kaggle/{type}_log.csv")

    data_dfs = [build_data_df(type, split) for split in sorted(log_df["split"].unique())]
    concat_data_df = pd.concat(data_dfs, ignore_index=True)

    df = log_df.merge(concat_data_df, on="object_id", how="left")

    df.drop(columns=["English Translation"], inplace=True)
    df.dropna(subset=["target"], inplace=True) if type == "train" else None

    return df

def build_data_df(type: Literal["train", "test"], split: str) -> pd.DataFrame:
    data_df = pd.read_csv(f"./resources/kaggle/{split}/{type}_full_lightcurves.csv")

    agg_data_df = data_df.groupby(["object_id", "Filter"]).agg(
        flux_mean=("Flux", "mean"),
        flux_std=("Flux", "std"),
        flux_min=("Flux", "min"),
        flux_max=("Flux", "max"),
        n_obs=("Flux", "count"),
    ).reset_index()

    pivot_data_df = agg_data_df.pivot(index="object_id", columns="Filter")
    pivot_data_df.columns = [f"{stat}_{filter}" for stat, filter in pivot_data_df.columns]
    pivot_data_df = pivot_data_df.reset_index()

    return pivot_data_df


if __name__ == "__main__":
    main()
