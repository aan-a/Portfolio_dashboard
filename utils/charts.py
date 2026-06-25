import plotly.express as px

def plot_price(df):
    fig = px.line(df, x="Date", y="Close", title="Stock Price Over Time")
    return fig


def plot_returns(df):
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()

    fig = px.bar(df, x="Date", y="Returns", title="Daily Returns")
    return fig