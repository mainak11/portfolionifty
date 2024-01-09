
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

class Nifty50Portfolio:
    def __init__(self, start_date, end_date):
        self.tickers = ['RELIANCE.NS', 'HCLTECH.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS', 'BAJFINANCE.NS', 'APOLLOHOSP.NS', 'WIPRO.NS', 'ADANIENT.NS']
        self.benchmark_ticker = '^NSEI'
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self, ticker):
        data = yf.download(ticker, start=self.start_date, end=self.end_date)
        return data['Adj Close']

    def create_portfolio(self):
        portfolio = pd.DataFrame(index=pd.date_range(start=self.start_date, end=self.end_date))

        for ticker in self.tickers:
            prices = self.download_data(ticker)
            portfolio[ticker] = prices

        return portfolio

    def active_stock_selection_strategy(self, portfolio):
        selected_stocks = portfolio.pct_change().mean(axis=1) > 0
        return selected_stocks

    def compare_with_benchmark(self, portfolio, benchmark_data):
        portfolio_returns = portfolio.pct_change().mean(axis=1).dropna()
        benchmark_returns = benchmark_data.pct_change().dropna()

        return portfolio_returns, benchmark_returns

    def summarize_performance(self, portfolio_returns, benchmark_returns):
        # Implement any performance metrics or analysis here
        cagr_portfolio = (portfolio_returns[-1] / portfolio_returns[0]) ** (1 / len(portfolio_returns.index.year.unique())) - 1
        cagr_benchmark = (benchmark_returns[-1] / benchmark_returns[0]) ** (1 / len(benchmark_returns.index.year.unique())) - 1

        # Calculate Volatility
        volatility_portfolio = np.sqrt(252) * portfolio_returns.std()
        volatility_benchmark = np.sqrt(252) * benchmark_returns.std()

        # Calculate Sharpe Ratio
        sharpe_ratio_portfolio = (np.sqrt(252) * portfolio_returns.mean()) / portfolio_returns.std()
        sharpe_ratio_benchmark = (np.sqrt(252) * benchmark_returns.mean()) / benchmark_returns.std()

        summary = pd.DataFrame({
            'Portfolio Returns': portfolio_returns.cumsum(),
            'Benchmark Returns': benchmark_returns.cumsum()
        })
        summary2 = pd.DataFrame({
            'CAGR': [cagr_portfolio * 100,cagr_benchmark * 100],
            'Volatility': [volatility_portfolio * 100,volatility_benchmark * 100],
            'Sharpe Ratio': [sharpe_ratio_portfolio,sharpe_ratio_benchmark],
        },index=['portfolio','benchmark'])
        return summary,summary2

    def plot_performance(self, summary):
        fig, ax = plt.subplots(figsize=(10, 6))
        performance_summary[['Portfolio Returns', 'Benchmark Returns']].plot(ax=ax)
        plt.title('Portfolio vs Benchmark Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend(['Stategy', 'Benchmark'])
        return fig

st.set_page_config(page_title="portfolio out of Nifty50 Stocks", page_icon="📊")

# Title of the app
st.title("Portfolio out of Nifty50 Stocks")

# Date input widgets
start_date = st.date_input("Select Start Date", pd.to_datetime("2019-01-01"))
end_date = st.date_input("Select End Date", pd.to_datetime("2022-12-31"))

nifty_portfolio = Nifty50Portfolio(start_date, end_date)

# Create Portfolio
portfolio_data = nifty_portfolio.create_portfolio()

# Download Benchmark Data
benchmark_data = nifty_portfolio.download_data(nifty_portfolio.benchmark_ticker)

# Implement Active Stock Selection Strategy
selected_stocks = nifty_portfolio.active_stock_selection_strategy(portfolio_data)

# Compare with Benchmark
portfolio_returns, benchmark_returns = nifty_portfolio.compare_with_benchmark(portfolio_data[selected_stocks], benchmark_data)

# Summarize Performance
performance_summary,df= nifty_portfolio.summarize_performance(portfolio_returns, benchmark_returns)

# Display the DataFrame
st.subheader("Generated Summary:")
st.dataframe(df)

# Display the Graph
st.subheader("Generated Graph:")
fig = nifty_portfolio.plot_performance(performance_summary)
st.pyplot(fig)
