from turtle import st

import pandas as pd

def load_data():
    df = pd.read_csv("data/stocks.csv")

    # basic cleaning
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Company", "Date"])

    return df