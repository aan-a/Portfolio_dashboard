import pandas as pd

def compute_metrics(df):
    df = df.copy()

    df["Daily Return"] = df["Close"].pct_change()

    metrics = {
        "Avg Price": df["Close"].mean(),
        "Max Price": df["Close"].max(),
        "Min Price": df["Close"].min(),
        "Volatility": df["Daily Return"].std(),
        "Total Return %": ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
    }

    return metrics