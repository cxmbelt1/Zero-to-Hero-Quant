import os
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def calculate_volatility(data):
    data["log_returns"] = np.log(data["Close"]/data["Close"].shift(1))

    data["Rolling_Std"] = data["log_returns"].rolling(window=20).std()
    data["Annualized_Vol"] = data["Rolling_Std"] * np.sqrt(252)
    
    data =data.dropna()
    return data

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == 'call':
        return S*norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r*T) * norm.cdf(-d2) - S*norm.cdf(-d1)
    else:
        return "Invalid contract type"

def calculate_option_price(data, risk_free_rate=0.03,days_till_expiration=30):
    
    T = days_till_expiration/252
    data = data.dropna().copy()

    #call price column
    data.loc[:, "Call_Price"] = data.apply(
        lambda row: black_scholes(
        S = row["Close"],
        K = row["Close"]+1,
        T = T,
        r = risk_free_rate,
        sigma=row["Annualized_Vol"],
        option_type='call'
        ), axis=1
    )
    
    #put price column
    data.loc[:, "Put_Price"] = data.apply(
        lambda row: black_scholes(
        S = row["Close"],
        K = row["Close"]-1,
        T = T,
        r = risk_free_rate,
        sigma=row["Annualized_Vol"],
        option_type='put'
        ), axis=1
    )

    return data

def plot_with_options(data,ticker):

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Candlestick Chart", "Call premium", "Put premium")
    )

    fig.add_trace(go.Candlestick(
        x = data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="Price"
        ), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["Call_Price"], 
                             mode='lines', name='Call Price', line=dict(color="green")), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data["Put_Price"], 
                             mode='lines', name='Put Price', line=dict(color="red")), row=3, col=1)    

    fig.update_layout(
        title = f"{ticker} Stock Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis3_title="Date",
        yaxis1_title="Price",
        yaxis2_title="Call_Price",
        yaxis3_title="Put_Price",
        height=900,
        showlegend=True
    )
    
    fig.show()


if __name__ == '__main__':
    
    tickers = ['F']
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    stock_data = get_stock_data(tickers=tickers,start=start_date,end=end_date)
    for ticker,data in stock_data.items():
        data = calculate_volatility(data)
        data = calculate_option_price(data)
        
        plot_with_options(data,ticker)