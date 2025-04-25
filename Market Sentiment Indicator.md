# Composite Market Sentiment Analyzer

A Python tool that calculates market sentiment by combining multiple financial indicators. Processes data from Yahoo Finance and FRED to produce a sentiment score from 0-100.

## Features

- **8 Market Indicators**: Momentum, market breadth, new highs/lows, options sentiment, credit spreads, volatility, safe haven flows, and economic data
- **Data Caching**: Smart caching and parallel processing for improved performance
- **Visualization**: Clean gauge chart displaying current sentiment
- **Reporting**: Optional markdown reports with market analysis

## Interpretation

- 0-20: Extremely Bearish
- 20-40: Bearish
- 40-60: Neutral
- 60-80: Bullish
- 80-100: Extremely Bullish

Check FRED API KEY
