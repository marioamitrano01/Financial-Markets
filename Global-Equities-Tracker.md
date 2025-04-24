# Global Stock Indices Tracker

Real-time dashboard for monitoring major global stock indices with technical indicators and market analysis.

## Features

- **Global Coverage**: Track 9 major indices (S&P 500, NASDAQ, Dow, DAX, Nikkei, etc.)
- **Technical Analysis**: EMAs, Bollinger Bands, RSI, Stochastic Oscillators
- **Market Regimes**: Auto-detection of trends, golden/death crosses
- **Performance**: YTD, 1M, 3M, 6M, 1Y returns
- **Correlation**: Dynamic cross-market correlation matrix
- **Multiple Views**: Basic, Technical, Investing, and Correlation modes
- **Auto-Refresh**: Optional 5-minute data updates


## Code Structure

- **Data Fetching**: yfinance with caching and error handling
- **Analysis Engine**: Technical indicators and market regime detection
- **Visualization**: Interactive Plotly charts with annotations
- **UI**: Responsive Streamlit interface with tabs and expandable sections

## Dependencies

streamlit, yfinance, pandas, numpy, plotly, ta

## Usage

1. Select time period/custom date range
2. Choose view mode (Basic/Technical/Investing/Correlation)
3. Optionally enable auto-refresh
