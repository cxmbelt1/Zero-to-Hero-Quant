import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_stock_data(tickers,start,end):
    data = {}
    output_dir = os.path.join(os.getcwd(), "CSVs")
    for ticker in tickers:
        try:

            file_path = os.path.join(output_dir, f"{ticker}_returns.csv")
            df = pd.read_csv(file_path,index_col=0,parse_dates=True)
            data[ticker] = df
        except Exception as e:
            print(f"Error: {e}")
            
            df = yf.download(ticker, start=start, end=end)
            data[ticker] = df
            df.to_csv(file_path)
    
    return data

def plot(data,ticker):

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,open=data['Open'],high=data['High'],
        low=data['Low'],close=data['Close'],name="Price"
        ))

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(rangeslider=dict(visible=False))
    )

    fig.show()

if __name__ == '__main__':
    
    tickers = ['AAPL', 'GOOG', 'MSFT'] #
    start = '2023-01-01'
    end = '2024-11-01'

    stock_data = get_stock_data(tickers=tickers,start=start,end=end)
    for ticker,data in stock_data.items():
        plot(data,ticker)