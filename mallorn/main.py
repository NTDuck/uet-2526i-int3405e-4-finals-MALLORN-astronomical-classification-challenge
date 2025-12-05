import pandas as pd
from autogluon import TabularDataset, TabularPredictor

def main():
    train_df = TabularDataset("./resources/kaggle/train_log.csv")
    test_df = TabularDataset("./resources/kaggle/test_log.csv")

    predictor = TabularPredictor(label="target").fit(train_df)

    


if __name__ == '__main__':
    main()
