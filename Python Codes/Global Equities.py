import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

st.set_page_config(
    page_title="Global Equity Tracker",
    page_icon="📈",
    layout="wide"
)

try:
    st.cache_data
    USE_NEW_CACHE = True
except AttributeError:
    USE_NEW_CACHE = False

def cached_func(ttl=300):
    if USE_NEW_CACHE:
        return st.cache_data(ttl=ttl)
    else:
        return st.cache(ttl=ttl, allow_output_mutation=True, suppress_st_warning=True)

def clear_cache():
    if USE_NEW_CACHE:
        st.cache_data.clear()
    else:
        st.cache.clear()

def rerun_app():
    if USE_NEW_CACHE:
        st.rerun()
    else:
        st.experimental_rerun()

st.markdown("""
<style>
 .main {
 padding: 1rem;
 }
 .stTabs [data-baseweb="tab-list"] {
 gap: 1px;
 }
 .stTabs [data-baseweb="tab"] {
 height: 50px;
 white-space: pre-wrap;
 background-color: #f0f2f6;
 border-radius: 4px 4px 0 0;
 gap: 1px;
 padding-top: 10px;
 padding-bottom: 10px;
 }
 .stTabs [aria-selected="true"] {
 background-color: #4c9cf1;
 color: white;
 }
</style>
""", unsafe_allow_html=True)

st.title("Global Stock Indices Tracker")
st.markdown("Real-time monitoring of major global stock indices with technical indicators for asset managers.")

indices = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
    "Nifty 50": "^NSEI",
    "FTSE MIB": "FTSEMIB.MI",
    "FTSE 100": "^FTSE",
    "CAC 40": "^FCHI"
}

def safe_download(ticker, period="1y", start=None, retries=3, delay=1):
    for i in range(retries):
        try:
            import inspect
            download_params = inspect.signature(yf.download).parameters
            supports_headers = 'headers' in download_params
            if start:
                if supports_headers:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    data = yf.download(ticker, start=start, progress=False, headers=headers)
                else:
                    data = yf.download(ticker, start=start, progress=False)
            else:
                if supports_headers:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    data = yf.download(ticker, period=period, progress=False, headers=headers)
                else:
                    data = yf.download(ticker, period=period, progress=False)
            if not data.empty:
                return data
            time.sleep(delay)
        except Exception as e:
            print(f"Attempt {i+1} failed for {ticker}: {e}")
            time.sleep(delay)
    print(f"All attempts failed for {ticker}, returning empty DataFrame")
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

def ensure_1d(series):
    if hasattr(series, 'iloc') and hasattr(series.iloc[0], 'iloc'):
        return series.iloc[:, 0]
    return series

def calculate_market_regime(df):
    if len(df) < 100 or df['SMA25'].isna().all() or df['SMA100'].isna().all():
        return "Insufficient Data"
    sma25 = df['SMA25'].dropna().iloc[-1]
    sma100 = df['SMA100'].dropna().iloc[-1]
    if sma25 > sma100:
        if len(df) > 20 and df['SMA25'].dropna().iloc[-20] < df['SMA100'].dropna().iloc[-20]:
            return "Strong Bullish (Golden Cross)"
        return "Bullish"
    else:
        if len(df) > 20 and df['SMA25'].dropna().iloc[-20] > df['SMA100'].dropna().iloc[-20]:
            return "Strong Bearish (Death Cross)"
        return "Bearish"

def calculate_momentum(df):
    if df['RSI'].isna().all():
        return "Insufficient Data"
    rsi = df['RSI'].dropna().iloc[-1]
    if len(df) > 20:
        recent_price_change = df['Normalized_Price'].iloc[-1] / df['Normalized_Price'].iloc[-20] - 1
    else:
        recent_price_change = 0
    if rsi > 70:
        if recent_price_change > 0.05:
            return "Very Strong (Possibly Overbought)"
        return "Strong"
    elif rsi > 60:
        return "Moderately Strong"
    elif rsi > 40:
        return "Neutral"
    elif rsi > 30:
        return "Moderately Weak"
    else:
        if recent_price_change < -0.05:
            return "Very Weak (Possibly Oversold)"
        return "Weak"

def calculate_volatility_regime(df):
    if df['Volatility_30d'].isna().all() or df['BB_Width'].isna().all():
        return "Insufficient Data"
    vol_30d = df['Volatility_30d'].dropna().iloc[-1]
    bb_width = df['BB_Width'].dropna().iloc[-1]
    if len(df) > 60:
        avg_bb_width = df['BB_Width'].dropna().iloc[-60:].mean()
        bb_width_ratio = bb_width / avg_bb_width
    else:
        bb_width_ratio = 1.0
    if vol_30d > 0.25:
        if bb_width_ratio > 1.3:
            return "Extremely High"
        return "High"
    elif vol_30d > 0.15:
        if bb_width_ratio > 1.2:
            return "Elevated"
        return "Moderate"
    elif vol_30d > 0.08:
        return "Normal"
    else:
        if bb_width_ratio < 0.7:
            return "Extremely Low"
        return "Low"

def create_correlation_matrix(start_date=None):
    if not start_date:
        start_date = "2020-01-01"
    st.info(f"Calculating correlation matrix from {start_date}")
    all_returns = pd.DataFrame()
    for name, ticker in indices.items():
        try:
            df = safe_download(ticker, start=start_date)
            if not df.empty:
                close_values = ensure_1d(df['Close'])
                returns = np.log(close_values / close_values.shift(1))
                all_returns[name] = returns
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    all_returns = all_returns.dropna()
    corr_matrix = all_returns.corr()
    return corr_matrix

def display_correlation_matrix(corr_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        colorbar=dict(title='Correlation')
    ))
    fig.update_layout(
        title="Correlation Matrix of Global Indices (Log Returns)",
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

@cached_func(ttl=300)
def fetch_index_data(ticker, period="1y", start=None):
    try:
        if start:
            df = safe_download(ticker, start=start)
        else:
            df = safe_download(ticker, period=period)
        if df.empty:
            return None
        close_values = ensure_1d(df['Close'])
        high_values = ensure_1d(df['High'])
        low_values = ensure_1d(df['Low'])
        df['Log_Return'] = np.log(close_values / close_values.shift(1))
        start_value = close_values.iloc[0]
        df['Normalized_Price'] = 100 * (close_values / start_value)
        for window in [25, 50, 100, 200]:
            df[f'SMA{window}'] = SMAIndicator(close=close_values, window=window).sma_indicator()
            df[f'EMA{window}'] = EMAIndicator(close=close_values, window=window).ema_indicator()
            df[f'Norm_SMA{window}'] = 100 * (df[f'SMA{window}'] / start_value)
            df[f'Norm_EMA{window}'] = 100 * (df[f'EMA{window}'] / start_value)
        rsi = RSIIndicator(close=close_values)
        df['RSI'] = rsi.rsi()
        bb = BollingerBands(close=close_values)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        df['Norm_BB_Upper'] = 100 * (df['BB_Upper'] / start_value)
        df['Norm_BB_Lower'] = 100 * (df['BB_Lower'] / start_value)
        df['Norm_BB_Mid'] = 100 * (df['BB_Mid'] / start_value)
        stoch = StochasticOscillator(high=high_values, low=low_values, close=close_values)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        df['Log_Return_Clean'] = df['Log_Return'].replace([np.inf, -np.inf], np.nan).dropna()
        trading_days_per_year = 252
        df['Volatility_30d'] = df['Log_Return_Clean'].rolling(window=30).std() * np.sqrt(trading_days_per_year)
        try:
            current_year = df.index[-1].year
            first_day_of_year = df[df.index.year == current_year].iloc[0].name
            df['YTD_Return'] = (close_values.iloc[-1] / close_values.loc[first_day_of_year]) - 1
        except Exception as e:
            print(f"Error calculating YTD return: {e}")
            df['YTD_Return'] = np.nan
        try:
            today = df.index[-1]
            one_month_ago = today - pd.Timedelta(days=30)
            nearest_1m = df.index[df.index <= one_month_ago].max() if any(df.index <= one_month_ago) else df.index[0]
            df['1M_Return'] = (close_values.iloc[-1] / close_values.loc[nearest_1m]) - 1
            three_months_ago = today - pd.Timedelta(days=90)
            nearest_3m = df.index[df.index <= three_months_ago].max() if any(df.index <= three_months_ago) else df.index[0]
            df['3M_Return'] = (close_values.iloc[-1] / close_values.loc[nearest_3m]) - 1
            six_months_ago = today - pd.Timedelta(days=180)
            nearest_6m = df.index[df.index <= six_months_ago].max() if any(df.index <= six_months_ago) else df.index[0]
            df['6M_Return'] = (close_values.iloc[-1] / close_values.loc[nearest_6m]) - 1
            one_year_ago = today - pd.Timedelta(days=365)
            nearest_1y = df.index[df.index <= one_year_ago].max() if any(df.index <= one_year_ago) else df.index[0]
            df['1Y_Return'] = (close_values.iloc[-1] / close_values.loc[nearest_1y]) - 1
        except Exception as e:
            print(f"Error calculating period returns: {e}")
            df['1M_Return'] = np.nan
            df['3M_Return'] = np.nan
            df['6M_Return'] = np.nan
            df['1Y_Return'] = np.nan
        current_price = float(df['Close'].iloc[-1].iloc[0]) if hasattr(df['Close'].iloc[-1], 'iloc') else float(df['Close'].iloc[-1])
        previous_price = float(df['Close'].iloc[-2].iloc[0]) if hasattr(df['Close'].iloc[-2], 'iloc') else float(df['Close'].iloc[-2])
        daily_change = (current_price - previous_price) / previous_price * 100
        try:
            ytd_return = float(df['YTD_Return'].iloc[-1]) * 100 if not pd.isna(df['YTD_Return'].iloc[-1]) else 0
        except Exception as e:
            print(f"Error processing YTD return: {e}")
            ytd_return = 0
        try:
            m1_return = float(df['1M_Return'].iloc[-1]) * 100 if not pd.isna(df['1M_Return'].iloc[-1]) else 0
        except Exception as e:
            print(f"Error processing 1M return: {e}")
            m1_return = 0
        try:
            m3_return = float(df['3M_Return'].iloc[-1]) * 100 if not pd.isna(df['3M_Return'].iloc[-1]) else 0
        except Exception as e:
            print(f"Error processing 3M return: {e}")
            m3_return = 0
        try:
            m6_return = float(df['6M_Return'].iloc[-1]) * 100 if not pd.isna(df['6M_Return'].iloc[-1]) else 0
        except Exception as e:
            print(f"Error processing 6M return: {e}")
            m6_return = 0
        try:
            y1_return = float(df['1Y_Return'].iloc[-1]) * 100 if not pd.isna(df['1Y_Return'].iloc[-1]) else 0
        except Exception as e:
            print(f"Error processing 1Y return: {e}")
            y1_return = 0
        try:
            vol_30d = float(df['Volatility_30d'].iloc[-1]) * 100 if not pd.isna(df['Volatility_30d'].iloc[-1]) else 0
        except Exception as e:
            print(f"Error processing volatility: {e}")
            vol_30d = 0
        return {
            'data': df,
            'current_price': current_price,
            'daily_change': daily_change,
            'last_updated': df.index[-1].strftime('%Y-%m-%d'),
            'performance': {
                'ytd': ytd_return,
                '1m': m1_return,
                '3m': m3_return,
                '6m': m6_return,
                '1y': y1_return,
                'volatility': vol_30d
            }
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def create_price_plot(data_dict, index_name, ma_periods=[25, 50, 100]):
    if data_dict is None:
        return go.Figure()
    df = data_dict['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Normalized_Price'],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    colors = ['orange', 'green', 'red', 'purple']
    for i, period in enumerate(ma_periods):
        if i < len(colors):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'Norm_EMA{period}'],
                mode='lines',
                name=f'EMA{period}',
                line=dict(color=colors[i])
            ))
    if 25 in ma_periods and 100 in ma_periods:
        if df['EMA25'].iloc[-1] > df['EMA100'].iloc[-1] and df['EMA25'].iloc[-20] < df['EMA100'].iloc[-20]:
            annotation_text = "Recent Golden Cross"
            annotation_color = "green"
        elif df['EMA25'].iloc[-1] < df['EMA100'].iloc[-1] and df['EMA25'].iloc[-20] > df['EMA100'].iloc[-20]:
            annotation_text = "Recent Death Cross"
            annotation_color = "red"
        elif df['EMA25'].iloc[-1] > df['EMA100'].iloc[-1]:
            annotation_text = "Bullish Trend"
            annotation_color = "green"
        else:
            annotation_text = "Bearish Trend"
            annotation_color = "red"
        fig.add_annotation(
            x=df.index[-1],
            y=df['Normalized_Price'].iloc[-1] * 1.05,
            text=annotation_text,
            showarrow=True,
            arrowhead=1,
            font=dict(color="white"),
            bgcolor=annotation_color,
            bordercolor=annotation_color
        )
    fig.update_layout(
        title=f"{index_name} - Price with EMAs",
        xaxis_title="Date",
        yaxis_title="Value (Normalized to 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_bollinger_plot(data_dict, index_name):
    if data_dict is None:
        return go.Figure()
    df = data_dict['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Normalized_Price'],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Norm_BB_Upper'],
        mode='lines',
        name='Upper Band',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Norm_BB_Mid'],
        mode='lines',
        name='Middle Band',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Norm_BB_Lower'],
        mode='lines',
        name='Lower Band',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Norm_BB_Upper'],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Norm_BB_Lower'],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(173, 216, 230, 0.2)',
        showlegend=False
    ))
    fig.update_layout(
        title=f"{index_name} - Price with Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Value (Normalized to 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_technical_plot(data_dict, index_name):
    if data_dict is None:
        return go.Figure()
    df = data_dict['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        mode='lines', name='RSI',
        line=dict(color='purple')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=[70] * len(df.index),
        mode='lines', name='Overbought',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=[30] * len(df.index),
        mode='lines', name='Oversold',
        line=dict(color='green', dash='dash')
    ))
    fig.update_layout(
        title=f"{index_name} - RSI Indicator",
        xaxis_title="Date",
        yaxis_title="RSI",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_stochastic_plot(data_dict, index_name):
    if data_dict is None:
        return go.Figure()
    df = data_dict['data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Stoch_K'],
        mode='lines', name='%K',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Stoch_D'],
        mode='lines', name='%D',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=[80] * len(df.index),
        mode='lines', name='Overbought',
        line=dict(color='gray', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=[20] * len(df.index),
        mode='lines', name='Oversold',
        line=dict(color='gray', dash='dash')
    ))
    fig.update_layout(
        title=f"{index_name} - Stochastic Oscillator",
        xaxis_title="Date",
        yaxis_title="Value",
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def display_correlation_view(time_period, custom_start, start_date):
    st.subheader("Correlation Matrix of Global Indices")
    corr_start_date = determine_start_date(time_period, custom_start, start_date)
    with st.spinner(f"Calculating correlation matrix from {corr_start_date}..."):
        corr_matrix = create_correlation_matrix(start_date=corr_start_date)
    st.plotly_chart(display_correlation_matrix(corr_matrix), use_container_width=True)
    with st.expander("View Raw Correlation Data"):
        st.dataframe(corr_matrix)
    st.markdown("""
    ### About Correlation Matrix
    This matrix shows the correlation between different global indices based on daily log returns for the selected time period.
    - **1.0**: Perfect positive correlation (indices move exactly together)
    - **0.0**: No correlation (indices move independently)
    - **-1.0**: Perfect negative correlation (indices move in opposite directions)
    Higher correlation (closer to 1.0) indicates similar market behavior, while lower values suggest diversification benefits.
    The correlation is calculated using logarithmic returns, which are considered more appropriate for financial time series analysis than simple percentage returns.
    """)

def determine_start_date(time_period, custom_start, start_date):
    if custom_start and start_date:
        return start_date.strftime('%Y-%m-%d')
    elif time_period != "max" and time_period != "ytd":
        today = datetime.now()
        if time_period == "1mo":
            return (today - timedelta(days=30)).strftime('%Y-%m-%d')
        elif time_period == "3mo":
            return (today - timedelta(days=90)).strftime('%Y-%m-%d')
        elif time_period == "6mo":
            return (today - timedelta(days=180)).strftime('%Y-%m-%d')
        elif time_period == "1y":
            return (today - timedelta(days=365)).strftime('%Y-%m-%d')
        elif time_period == "2y":
            return (today - timedelta(days=730)).strftime('%Y-%m-%d')
        elif time_period == "5y":
            return (today - timedelta(days=1825)).strftime('%Y-%m-%d')
    else:
        return "2000-01-01" if time_period == "max" else datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')

def display_indices_views(tabs, time_period, custom_start, start_date, view_mode):
    for i, (index_name, ticker) in enumerate(indices.items()):
        with tabs[i]:
            if custom_start and start_date:
                data = fetch_index_data(ticker, start=start_date.strftime('%Y-%m-%d'))
            else:
                data = fetch_index_data(ticker, period=time_period)
            if data is None:
                st.error(f"Unable to fetch data for {index_name}")
                continue
            if view_mode == "Basic":
                display_basic_view(data, index_name, ticker)
            elif view_mode == "Technical Analysis":
                display_technical_view(data, index_name)
            else:
                display_investing_view(data, index_name)
            with st.expander("View Raw Data"):
                st.dataframe(data['data'].tail(100))

def display_basic_view(data, index_name, ticker):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Current Value",
            value=f"{data['current_price']:.2f}",
            delta=f"{data['daily_change']:.2f}%"
        )
    with col2:
        st.write(f"Last Updated: {data['last_updated']}")
    with col3:
        st.write(f"Ticker: {ticker}")
    st.plotly_chart(create_price_plot(data, index_name), use_container_width=True)
    st.plotly_chart(create_bollinger_plot(data, index_name), use_container_width=True)

def display_technical_view(data, index_name):
    cols = st.columns(4)
    cols[0].metric("Current", f"{data['current_price']:.2f}", f"{data['daily_change']:.2f}%")
    cols[1].metric("RSI", f"{data['data']['RSI'].iloc[-1]:.1f}",
                 "Overbought" if data['data']['RSI'].iloc[-1] > 70 else
                 "Oversold" if data['data']['RSI'].iloc[-1] < 30 else "Neutral")
    cols[2].metric("BB Width", f"{data['data']['BB_Width'].iloc[-1]:.2f}",
                 f"{data['data']['BB_Width'].iloc[-1] - data['data']['BB_Width'].iloc[-20]:.2f}")
    cols[3].metric("Stoch K/D", f"{data['data']['Stoch_K'].iloc[-1]:.1f}/{data['data']['Stoch_D'].iloc[-1]:.1f}")
    st.plotly_chart(create_bollinger_plot(data, index_name), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_technical_plot(data, index_name), use_container_width=True)
    with col2:
        st.plotly_chart(create_stochastic_plot(data, index_name), use_container_width=True)

def display_investing_view(data, index_name):
    perf = data['performance']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("YTD Return", f"{perf['ytd']:.2f}%")
        st.metric("1M Return", f"{perf['1m']:.2f}%")
    with col2:
        st.metric("3M Return", f"{perf['3m']:.2f}%")
        st.metric("6M Return", f"{perf['6m']:.2f}%")
    with col3:
        st.metric("1Y Return", f"{perf['1y']:.2f}%")
        st.metric("Volatility (30d)", f"{perf['volatility']:.2f}%")
    with col4:
        st.metric("Current", f"{data['current_price']:.2f}", f"{data['daily_change']:.2f}%")
        bb_position = ((data['current_price'] - data['data']['BB_Lower'].iloc[-1]) /
                     (data['data']['BB_Upper'].iloc[-1] - data['data']['BB_Lower'].iloc[-1]) * 100)
        st.metric("BB Position", f"{bb_position:.1f}%",
                f"{'Overbought' if bb_position > 80 else 'Oversold' if bb_position < 20 else 'Neutral'}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_price_plot(data, index_name, ma_periods=[25, 100]), use_container_width=True)
    with col2:
        st.plotly_chart(create_bollinger_plot(data, index_name), use_container_width=True)
    with st.expander("Technical Indicators"):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_technical_plot(data, index_name), use_container_width=True)
        with col2:
            st.plotly_chart(create_stochastic_plot(data, index_name), use_container_width=True)
    with st.expander("Market Regime Analysis"):
        df = data['data']
        market_regime = calculate_market_regime(df)
        momentum = calculate_momentum(df)
        volatility = calculate_volatility_regime(df)
        cols = st.columns(3)
        cols[0].metric("Market Regime", market_regime)
        cols[1].metric("Momentum", momentum)
        cols[2].metric("Volatility", volatility)

def handle_auto_refresh():
    time.sleep(1)
    st.write("Auto-refresh is enabled. Page will refresh every 5 minutes.")
    time_placeholder = st.empty()
    count = 300
    while count > 0:
        with time_placeholder:
            mins, secs = divmod(count, 60)
            st.write(f"Next refresh in: {mins:02d}:{secs:02d}")
        time.sleep(1)
        count -= 1
    if count == 0:
        clear_cache()
        rerun_app()

def main():
    tabs = st.tabs(list(indices.keys()))
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Refresh Data"):
            clear_cache()
            rerun_app()
    with col2:
        st.write(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    time_options = {
        "1mo": "1 month",
        "3mo": "3 months",
        "6mo": "6 months",
        "1y": "1 year",
        "2y": "2 years",
        "5y": "5 years",
        "ytd": "Year to date",
        "max": "Maximum available"
    }
    
    col1, col2 = st.columns([1, 1])
    with col1:
        time_period = st.selectbox("Select Time Period",
                                options=list(time_options.keys()),
                                format_func=lambda x: time_options[x],
                                index=3)
    with col2:
        custom_start = st.checkbox("Use custom start date")
        if custom_start:
            start_date = st.date_input("Start date",
                                    value=datetime(2020, 1, 1),
                                    min_value=datetime(2000, 1, 1),
                                    max_value=datetime.now())
        else:
            start_date = None
    
    auto_refresh = st.checkbox("Enable Auto-Refresh (every 5 minutes)")
    
    view_mode = st.radio("View Mode",
                        ["Basic", "Technical Analysis", "Investing View", "Correlation Matrix"],
                        horizontal=True)
    
    if view_mode == "Correlation Matrix":
        display_correlation_view(time_period, custom_start, start_date)
    else:
        display_indices_views(tabs, time_period, custom_start, start_date, view_mode)
    
    if auto_refresh:
        handle_auto_refresh()

if __name__ == "__main__":
    main()
