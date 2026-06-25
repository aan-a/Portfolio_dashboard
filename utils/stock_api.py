import yfinance as yf
import pandas as pd

def get_stock_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    df = df.reset_index()
    return df