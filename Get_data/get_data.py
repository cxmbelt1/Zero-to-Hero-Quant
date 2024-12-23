import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

if __name__ == '__main__':
    
    tickers = ['AAPL', 'GOOG', 'MSFT'] #
    start = '2023-01-01'
    end = '2024-11-01'