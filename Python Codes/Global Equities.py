import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

ALPHA_VANTAGE_API_KEY = "W7H26WB9J23S3R82"

class APICallCounter:
    def __init__(self, max_calls_per_minute=5, max_calls_per_day=500):
        self.calls_per_minute = 0
        self.calls_per_day = 0
        self.last_reset_minute = datetime.now().minute
        self.last_reset_day = datetime.now().day
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_day = max_calls_per_day
        
    def increment(self):
        current_time = datetime.now()
        
        if current_time.minute != self.last_reset_minute:
            self.calls_per_minute = 0
            self.last_reset_minute = current_time.minute
            
        if current_time.day != self.last_reset_day:
            self.calls_per_day = 0
            self.last_reset_day = current_time.day
            
        self.calls_per_minute += 1
        self.calls_per_day += 1
        
    def can_make_call(self):
        current_time = datetime.now()
        
        if current_time.minute != self.last_reset_minute:
            self.calls_per_minute = 0
            self.last_reset_minute = current_time.minute
            
        if current_time.day != self.last_reset_day:
            self.calls_per_day = 0
            self.last_reset_day = current_time.day
            
        return (self.calls_per_minute < self.max_calls_per_minute and 
                self.calls_per_day < self.max_calls_per_day)
                
    def get_stats(self):
        return {
            "minute": {
                "used": self.calls_per_minute,
                "max": self.max_calls_per_minute,
                "remaining": self.max_calls_per_minute - self.calls_per_minute
            },
            "day": {
                "used": self.calls_per_day,
                "max": self.max_calls_per_day,
                "remaining": self.max_calls_per_day - self.calls_per_day
            }
        }
        
    def wait_if_needed(self):
        if self.calls_per_minute >= self.max_calls_per_minute:
            current_time = datetime.now()
            seconds_to_next_minute = 60 - current_time.second
            return seconds_to_next_minute
        return 0

api_counter = APICallCounter()

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
    "S&P 500": "SPY", 
    "NASDAQ": "QQQ",   
    "Dow Jones": "DIA",  
    "DAX": "DAX",
    "Nikkei 225": "NKY",
    "FTSE 100": "UKX",
    "CAC 40": "CAC",
    "FTSE MIB": "FTSEMIB.MI",
    "Nifty 50": "NSEI"
}

@cached_func(ttl=3600)  
def fetch_alpha_vantage_data(symbol, function="TIME_SERIES_DAILY_ADJUSTED", outputsize="full"):
    """Fetch data from Alpha Vantage API with rate limit handling"""
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    for attempt in range(3):  
        wait_time = api_counter.wait_if_needed()
        if wait_time > 0:
            st.warning(f"API rate limit reached. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
        try:
            response = requests.get(url, params=params)
            api_counter.increment()
            
            data = response.json()
            
            if "Note" in data and "API call frequency" in data["Note"]:
                wait_time = 60 
                st.warning(f"Alpha Vantage rate limit reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            if "Error Message" in data:
                print(f"Alpha Vantage API Error: {data['Error Message']}")
                time.sleep(2)  
                continue
                
            key_name = "Time Series (Daily)"
            if function == "TIME_SERIES_DAILY_ADJUSTED":
                key_name = "Time Series (Daily)"
            elif function == "TIME_SERIES_WEEKLY_ADJUSTED":
                key_name = "Weekly Adjusted Time Series"
            elif function == "TIME_SERIES_MONTHLY_ADJUSTED":
                key_name = "Monthly Adjusted Time Series"
                
            if key_name not in data:
                print(f"Unexpected response format: {data.keys()}")
                time.sleep(2)
                continue
                
            time_series = data[key_name]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adj Close',
                '6. volume': 'Volume'
            })
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            df.index = pd.to_datetime(df.index)
            
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {symbol}: {str(e)}")
            time.sleep(2)  
    
    print(f"All attempts failed for {symbol}, returning empty DataFrame")
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

def create_correlation_matrix(start_date=None, data_function="TIME_SERIES_DAILY_ADJUSTED"):
    if not start_date:
        start_date = "2020-01-01"
    st.info(f"Calculating correlation matrix from {start_date}")
    
    all_returns = pd.DataFrame()
    for name, ticker in indices.items():
        try:
            df = fetch_alpha_vantage_data(ticker, function=data_function)
            if not df.empty:
                df = df[df.index >= start_date]
                
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
def fetch_index_data(ticker, period="1y", start=None, data_function="TIME_SERIES_DAILY_ADJUSTED"):
    try:
        df = fetch_alpha_vantage_data(ticker, function=data_function)
        
        if df.empty:
            return None
            
        if start:
            df = df[df.index >= start]
        elif period != "max":
            today = datetime.now()
            if period == "1mo":
                start_date = today - timedelta(days=30)
            elif period == "3mo":
                start_date = today - timedelta(days=90)
            elif period == "6mo":
                start_date = today - timedelta(days=180)
            elif period == "1y":
                start_date = today - timedelta(days=365)
            elif period == "2y":
                start_date = today - timedelta(days=730)
            elif period == "5y":
                start_date = today - timedelta(days=1825)
            elif period == "ytd":
                start_date = datetime(today.year, 1, 1)
            else:
                start_date = today - timedelta(days=365)  
                
            df = df[df.index >= start_date]
        
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
        
        current_price = float(close_values.iloc[-1])
        previous_price = float(close_values.iloc[-2])
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
def determine_start_date(time_period, custom_start, start_date):
    if custom_start and start_date:
        return start_date.strftime('%Y-%m-%d')
    elif time_period != "max" and time_period != "ytd":
        today = datetime.now()
        if time_period == "1mo":
            return (today - timedelta(days=30)).strftime('%Y-%m-%d')
        elif time_period == "3mo":
            return (today - timedelta(days=90)).strftime('%Y-%m-%d')
    else:
        return "2000-01-01" if time_period == "max" else datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')


def display_correlation_view(time_period, custom_start, start_date, data_frequency):
    st.subheader("Correlation Matrix of Global Indices")
    corr_start_date = determine_start_date(time_period, custom_start, start_date)
    
    api_function = "TIME_SERIES_DAILY_ADJUSTED"
    if data_frequency == "Weekly":
        api_function = "TIME_SERIES_WEEKLY_ADJUSTED"
    elif data_frequency == "Monthly":
        api_function = "TIME_SERIES_MONTHLY_ADJUSTED"
    
    with st.spinner(f"Calculating correlation matrix from {corr_start_date}..."):
        corr_matrix = create_correlation_matrix(start_date=corr_start_date, data_function=api_function)
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
