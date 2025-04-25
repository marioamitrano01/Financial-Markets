import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from fredapi import Fred
import datetime
import warnings
import argparse
import logging
import os
import json
from functools import lru_cache
from math import erf, sqrt
from scipy.stats import norm
from typing import Tuple, List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("market_sentiment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("market_sentiment")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class DataCache:
  
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"
    
    def exists(self, key: str) -> bool:
        path = self.get_cache_path(key)
        return path.exists()
    
    def get(self, key: str) -> Optional[Any]:
        path = self.get_cache_path(key)
        if path.exists():
            try:
                return pd.read_pickle(path)
            except Exception as e:
                logger.warning(f"Failed to read cache for {key}: {e}")
                return None
        return None
    
    def set(self, key: str, data: Any) -> None:
        path = self.get_cache_path(key)
        try:
            pd.to_pickle(data, path)
        except Exception as e:
            logger.warning(f"Failed to write cache for {key}: {e}")


@lru_cache(maxsize=1)
def get_nasdaq_tickers() -> List[str]:
    
    cache = DataCache()
    cache_key = "nasdaq_tickers"
    
    cached_tickers = cache.get(cache_key)
    if cached_tickers is not None:
        logger.info("Using cached NASDAQ tickers")
        return cached_tickers
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching NASDAQ tickers (attempt {attempt+1}/{max_retries})")
            
            tables = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
            for table in tables:
                if "Ticker" in table.columns:
                    tickers = table["Ticker"].tolist()
                    cleaned_tickers = [ticker.strip() for ticker in tickers if isinstance(ticker, str)]
                    
                    if cleaned_tickers:
                        cache.set(cache_key, cleaned_tickers)
                        return cleaned_tickers
            
            url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=100&offset=0&exchange=NASDAQ"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and "rows" in data["data"]:
                    tickers = [row["symbol"] for row in data["data"]["rows"] if row.get("symbol")]
                    if tickers:
                        cache.set(cache_key, tickers)
                        return tickers
            
            logger.warning("Using fallback NASDAQ tickers list")
            fallback_tickers = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "PYPL", 
                "INTC", "CMCSA", "PEP", "CSCO", "ADBE", "NFLX", "AVGO", "TXN", 
                "QCOM", "COST", "TMUS", "CHTR", "SBUX", "MDLZ", "AMAT", "AMD", 
                "BKNG", "GILD", "ISRG", "INTU", "ADP", "VRTX", "REGN", "ILMN"
            ]
            cache.set(cache_key, fallback_tickers)
            return fallback_tickers
            
        except Exception as e:
            logger.error(f"Error fetching NASDAQ tickers (attempt {attempt+1}): {e}")
            time.sleep(2)  
    
    logger.error("All attempts to fetch NASDAQ tickers failed")
    default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
    return default_tickers


class DataManager:
    
    def __init__(self, start_date: str, end_date: str, fred_api_key: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.fred = Fred(api_key=fred_api_key)
        self.cache = DataCache()
        self.session = requests.Session()
        self._setup_session()
        
    def _setup_session(self):
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def get_history(self, ticker: str) -> pd.DataFrame:
        
        cache_key = f"history_{ticker}_{self.start_date}_{self.end_date}"
        
        cached_data = self.cache.get(cache_key)
        if cached_data is not None and not cached_data.empty:
            return cached_data
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)
                if not data.empty:
                    self.cache.set(cache_key, data)
                    return data
                time.sleep(1)  
            except Exception as e:
                logger.warning(f"Error fetching history for {ticker} (attempt {attempt+1}): {e}")
                time.sleep(2) 
        
        logger.error(f"All attempts to fetch history for {ticker} failed")
        return pd.DataFrame()

    def get_fred_series(self, series_code: str) -> pd.DataFrame:
        
        cache_key = f"fred_{series_code}_{self.start_date}_{self.end_date}"
        
        cached_data = self.cache.get(cache_key)
        if cached_data is not None and not cached_data.empty:
            return cached_data
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = self.fred.get_series(series_code)
                if data is not None and not data.empty:
                    df = pd.DataFrame(data, columns=[series_code]).dropna()
                    df.index = pd.to_datetime(df.index)
                    mask = (df.index >= pd.to_datetime(self.start_date)) & (df.index <= pd.to_datetime(self.end_date))
                    result = df.loc[mask]
                    if not result.empty:
                        self.cache.set(cache_key, result)
                        return result
                time.sleep(1)  
            except Exception as e:
                logger.warning(f"Error fetching FRED series {series_code} (attempt {attempt+1}): {e}")
                time.sleep(2)  
        
        logger.error(f"All attempts to fetch FRED series {series_code} failed")
        return pd.DataFrame()
    
    def fetch_batch_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
            future_to_ticker = {executor.submit(self.get_history, ticker): ticker for ticker in tickers}
            for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Fetching stock data"):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[ticker] = data
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
        
        return results


class Indicator:
    
    def __init__(self, data_manager: DataManager) -> None:
        self.dm = data_manager
        self.name = self.__class__.__name__.replace("Indicator", "")
    
    def calculate(self) -> float:
        raise NotImplementedError("Must implement calculate method")
    
    def get_historical_values(self) -> List[float]:
        return []
    
    @staticmethod
    def scale_with_history(values: np.ndarray, current: float) -> float:
        
        values = np.array(values)
        if len(values) < 10:
            return Indicator.robust_percentile(values, current)
        
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
        
        if len(filtered_values) < len(values) * 0.7:
            filtered_values = values
            
        m = np.mean(filtered_values)
        s = np.std(filtered_values)
        
        if s < 1e-8:
            return 50.0
            
        z = (current - m) / s
        sc = norm.cdf(z) * 100
        return max(0, min(100, sc))

    @staticmethod
    def robust_percentile(series: np.ndarray, val: float) -> float:
        
        if len(series) < 5:
            return 50.0
        return (np.sum(series < val) / len(series)) * 100


class MomentumIndicator(Indicator):
    
    def calculate(self) -> float:
        spy = self.dm.get_history("SPY")
        if spy.empty or len(spy) < 125:
            logger.warning("Insufficient SPY data for momentum calculation")
            return 50.0
            
        spy["SMA125"] = spy["Close"].rolling(window=125).mean()
        spy = spy.dropna()
        
        if spy.empty:
            return 50.0
            
        current_price = spy["Close"].iloc[-1]
        sma = spy["SMA125"].iloc[-1]
        momentum_current = (current_price - sma) / sma
        
        momentum_series = ((spy["Close"] - spy["SMA125"]) / spy["SMA125"]).dropna().values
        
        return Indicator.scale_with_history(momentum_series, momentum_current)
        
    def get_historical_values(self) -> List[float]:
        spy = self.dm.get_history("SPY")
        if spy.empty or len(spy) < 125:
            return []
            
        spy["SMA125"] = spy["Close"].rolling(window=125).mean()
        spy = spy.dropna()
        
        if spy.empty:
            return []
            
        spy["Momentum"] = ((spy["Close"] - spy["SMA125"]) / spy["SMA125"]) * 100
        
        values = []
        for i in range(125, len(spy)):
            current = spy["Momentum"].iloc[i]
            history = spy["Momentum"].iloc[max(0, i-365):i].values
            if len(history) >= 30: 
                scaled = Indicator.scale_with_history(history, current)
                values.append(scaled)
            else:
                values.append(50.0)
                
        return values[-90:]  


class NewHighsLowsIndicator(Indicator):
    
    def calculate(self) -> float:
        tickers = get_nasdaq_tickers()
        if not tickers:
            logger.warning("No tickers available for NewHighsLows calculation")
            return 50.0
            
        stock_data = self.dm.fetch_batch_stock_data(tickers)
        
        if not stock_data:
            logger.warning("No stock data available for NewHighsLows calculation")
            return 50.0
            
        ratios = []
        for ticker, df in stock_data.items():
            if df.empty or len(df) < 30:
                continue
                
            period_end = df.index[-1]
            half_year_date = period_end - pd.Timedelta(days=182)
            recent_df = df[df.index >= half_year_date]
            
            if recent_df.empty:
                continue
                
            cond_half = 1 if (recent_df["Close"].iloc[-1] >= recent_df["Close"].max() * 0.98) else 0
            
            cond_full = 1 if df["Close"].iloc[-1] >= df["Close"].max() * 0.98 else 0
            
            ratios.append((cond_half * 0.7 + cond_full * 0.3))
        
        if not ratios:
            logger.warning("No valid stocks for NewHighsLows calculation")
            return 50.0
            
        raw = np.mean(ratios)
        
        real_history = None 
        
        if real_history is not None:
            return Indicator.scale_with_history(real_history, raw)
        else:
            pseudo_history = np.random.beta(3, 3, 100) * 0.8 + 0.1  
            return Indicator.scale_with_history(pseudo_history, raw) 


class MarketBreadthIndicator(Indicator):
    
    def calculate(self) -> float:
        tickers = get_nasdaq_tickers()
        if not tickers:
            logger.warning("No tickers available for MarketBreadth calculation")
            return 50.0
            
        stock_data = self.dm.fetch_batch_stock_data(tickers)
        
        if not stock_data:
            logger.warning("No stock data available for MarketBreadth calculation")
            return 50.0
            
        scores = []
        for ticker, df in stock_data.items():
            if df.empty or len(df) < 50:
                continue
                
            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df["SMA200"] = df["Close"].rolling(window=200).mean()
            df = df.dropna()
            
            if df.empty:
                continue
                
            current_price = df["Close"].iloc[-1]
            
            above_20 = 1 if current_price > df["SMA20"].iloc[-1] else 0
            above_50 = 1 if current_price > df["SMA50"].iloc[-1] else 0
            above_200 = 1 if current_price > df["SMA200"].iloc[-1] else 0
            
            score = (above_20 * 0.2) + (above_50 * 0.3) + (above_200 * 0.5)
            scores.append(score * 100)
        
        if not scores:
            logger.warning("No valid stocks for MarketBreadth calculation")
            return 50.0
            
        raw = np.mean(scores)
        
        historical_breadth = None  
        
        if historical_breadth is not None:
            return Indicator.scale_with_history(historical_breadth, raw)
        else:
            pseudo_history = np.random.beta(5, 5, 100) * 100
            return Indicator.scale_with_history(pseudo_history, raw)


class PutCallIndicator(Indicator):
    
    def calculate(self) -> float:
        try:
            spy = yf.Ticker("SPY")
            
            if not spy.options or len(spy.options) == 0:
                logger.warning("No options data available for SPY")
                return 50.0
                
            expirations = spy.options[:min(3, len(spy.options))]
            
            total_put_vol = 0
            total_call_vol = 0
            
            for expiration in expirations:
                try:
                    chain = spy.option_chain(expiration)
                    put_vol = chain.puts["volume"].sum()
                    call_vol = chain.calls["volume"].sum()
                    
                    total_put_vol += put_vol
                    total_call_vol += call_vol
                except Exception as e:
                    logger.error(f"Error fetching options for expiration {expiration}: {e}")
            
            if total_call_vol <= 0:
                logger.warning("No valid call volume for Put/Call calculation")
                return 50.0
                
            ratio = total_put_vol / total_call_vol
            
            score = 100 / (1 + np.exp((ratio - 0.85) * 5))
            
            return score
            
        except Exception as e:
            logger.error(f"Error computing Put/Call indicator: {e}")
            return 50.0


class JunkBondIndicator(Indicator):
    
    def calculate(self) -> float:
        baa_df = self.dm.get_fred_series("BAA")
        treas_df = self.dm.get_fred_series("DGS10")
        
        if baa_df.empty or treas_df.empty:
            logger.warning("Missing data for JunkBond calculation")
            return 50.0
            
        common_dates = baa_df.index.intersection(treas_df.index)
        
        if len(common_dates) == 0:
            logger.warning("No common dates for JunkBond calculation")
            return 50.0
            
        last_date = common_dates[-1]
        baa_val = baa_df.loc[last_date, "BAA"]
        treas_val = treas_df.loc[last_date, "DGS10"]
        
        spread_current = baa_val - treas_val
        
        joined = baa_df.join(treas_df, how="inner")
        joined["spread"] = joined["BAA"] - joined["DGS10"]
        spread_history = joined["spread"].dropna().values
        
        scaled = Indicator.scale_with_history(spread_history, spread_current)
        
        return max(0, min(100, 100 - scaled))
        
    def get_historical_values(self) -> List[float]:
        baa_df = self.dm.get_fred_series("BAA")
        treas_df = self.dm.get_fred_series("DGS10")
        
        if baa_df.empty or treas_df.empty:
            return []
            
        joined = baa_df.join(treas_df, how="inner")
        joined["spread"] = joined["BAA"] - joined["DGS10"]
        
        values = []
        window = 365  
        
        for i in range(window, len(joined)):
            current = joined["spread"].iloc[i]
            history = joined["spread"].iloc[i-window:i].values
            scaled = 100 - Indicator.scale_with_history(history, current)
            values.append(scaled)
            
        return values[-90:]  


class VIXIndicator(Indicator):
    
    def calculate(self) -> float:
        vix_df = self.dm.get_history("^VIX")
        
        if vix_df.empty:
            logger.warning("No VIX data available")
            return 50.0
            
        current_vix = vix_df["Close"].iloc[-1]
        vix_history = vix_df["Close"].dropna().values
        
        q99 = np.percentile(vix_history, 99)
        filtered_vix = vix_history[vix_history <= q99]
        
        scaled = Indicator.scale_with_history(filtered_vix, current_vix)
        
        return max(0, min(100, 100 - scaled))
        
    def get_historical_values(self) -> List[float]:
        vix_df = self.dm.get_history("^VIX")
        
        if vix_df.empty or len(vix_df) < 365:
            return []
            
        values = []
        window = 365  
        
        for i in range(window, len(vix_df)):
            current = vix_df["Close"].iloc[i]
            history = vix_df["Close"].iloc[i-window:i].values
            scaled = 100 - Indicator.scale_with_history(history, current)
            values.append(scaled)
            
        return values[-90:]  


class SafeHavenIndicator(Indicator):
    
    def calculate(self) -> float:
        tlt_df = self.dm.get_history("TLT")
        spy_df = self.dm.get_history("SPY")
        
        if tlt_df.empty or spy_df.empty:
            logger.warning("Missing data for SafeHaven calculation")
            return 50.0
            
        common = tlt_df.index.intersection(spy_df.index)
        
        if len(common) < 30:
            logger.warning("Insufficient data for SafeHaven calculation")
            return 50.0
            
        tlt_close = tlt_df.loc[common, "Close"]
        spy_close = spy_df.loc[common, "Close"]
        
        current_diff = (
            (tlt_close.iloc[-1] - tlt_close.iloc[-30]) / tlt_close.iloc[-30] -
            (spy_close.iloc[-1] - spy_close.iloc[-30]) / spy_close.iloc[-30]
        )
        
        diffs = []
        for i in range(30, len(common)):
            r_tlt = (tlt_close.iloc[i] - tlt_close.iloc[i-30]) / tlt_close.iloc[i-30]
            r_spy = (spy_close.iloc[i] - spy_close.iloc[i-30]) / spy_close.iloc[i-30]
            diffs.append(r_tlt - r_spy)
        
        diffs = np.array(diffs)
        
        scaled = Indicator.scale_with_history(diffs, current_diff)
        
        return max(0, min(100, 100 - scaled))


class EconomicIndicator(Indicator):
    
    def calculate(self) -> float:
        
        unrate = self.dm.get_fred_series("UNRATE")
        claims = self.dm.get_fred_series("ICSA")
        indpro = self.dm.get_fred_series("INDPRO")
        
        scores = []
        
        if not unrate.empty:
            current = unrate.iloc[-1, 0]
            history = unrate.iloc[:, 0].values
            scores.append(100 - Indicator.scale_with_history(history, current))
        
        if not claims.empty:
            current = claims.iloc[-1, 0]
            history = claims.iloc[:, 0].values
            scores.append(100 - Indicator.scale_with_history(history, current))
        
        if not indpro.empty:
            current = indpro.iloc[-1, 0]
            history = indpro.iloc[:, 0].values
            scores.append(Indicator.scale_with_history(history, current))
        
        if not scores:
            logger.warning("No economic data available")
            return 50.0
        
        return np.mean(scores)


class SentimentHistory:
    
    def __init__(self, file_path: str = "sentiment_history.json"):
        self.file_path = file_path
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, List[Dict[str, Any]]]:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading sentiment history: {e}")
        
        return {
            "composite": [],
            "indicators": {}
        }
    
    def save_history(self) -> None:
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sentiment history: {e}")
    
    def add_sentiment(self, date: str, composite: float, indicators: Dict[str, float]) -> None:
        self.history["composite"].append({
            "date": date,
            "value": composite
        })
        
        for name, value in indicators.items():
            if name not in self.history["indicators"]:
                self.history["indicators"][name] = []
            
            self.history["indicators"][name].append({
                "date": date,
                "value": value
            })
        
        max_history = 365
        for key in self.history["indicators"]:
            if len(self.history["indicators"][key]) > max_history:
                self.history["indicators"][key] = self.history["indicators"][key][-max_history:]
        
        if len(self.history["composite"]) > max_history:
            self.history["composite"] = self.history["composite"][-max_history:]
        
        self.save_history()


class CompositeSentiment:
    
    def __init__(self, data_manager: DataManager, weights: Dict[str, float] = None) -> None:
        self.dm = data_manager
        self.indicators = [
            MomentumIndicator(self.dm),
            NewHighsLowsIndicator(self.dm),
            MarketBreadthIndicator(self.dm),
            PutCallIndicator(self.dm),
            JunkBondIndicator(self.dm),
            VIXIndicator(self.dm),
            SafeHavenIndicator(self.dm),
            EconomicIndicator(self.dm) 
        ]
        
        default_weights = {
            "Momentum": 1.0,
            "NewHighsLows": 1.0,
            "MarketBreadth": 1.0,
            "PutCall": 0.8,
            "JunkBond": 1.0,
            "VIX": 1.0,
            "SafeHaven": 0.7,
            "Economic": 0.9
        }
        
        self.weights = weights if weights is not None else default_weights
        self.history = SentimentHistory()
    
    def compute(self) -> Tuple[float, Dict[str, float], Dict[str, List[float]]]:
        
        indicator_values = {}
        historical_values = {}
        
        for ind in self.indicators:
            try:
                name = ind.name
                value = ind.calculate()
                indicator_values[name] = value
                
                history = ind.get_historical_values()
                if history:
                    historical_values[name] = history
                    
                logger.info(f"{name:15s}: {round(value, 2)}")
            except Exception as e:
                logger.error(f"Error computing indicator {type(ind).__name__}: {e}")
                indicator_values[ind.name] = 50.0
        
        total_weight = 0
        weighted_sum = 0
        
        for name, value in indicator_values.items():
            weight = self.weights.get(name, 1.0)
            weighted_sum += value * weight
            total_weight += weight
        
        if total_weight > 0:
            composite = weighted_sum / total_weight
        else:
            composite = 50.0
            
        composite = max(0, min(100, composite))
        
        today = datetime.date.today().isoformat()
        self.history.add_sentiment(today, composite, indicator_values)
        
        return composite, indicator_values, historical_values


def gauge_plot(value: float, indicator_values: Dict[str, float] = None) -> go.Figure:
   
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},  
        title={'text': "Composite Sentiment", 'font': {'size': 24}},
        number={'font': {'size': 60}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))
    
    fig.update_layout(
        height=500,
        width=500,
        margin=dict(l=30, r=30, b=30, t=30)
    )
    
    return fig


def get_sentiment_rating(value: float) -> str:
    
    if value >= 80:
        return "Extremely Bullish"
    elif value >= 65:
        return "Bullish"
    elif value >= 55:
        return "Mildly Bullish"
    elif value >= 45:
        return "Neutral"
    elif value >= 35:
        return "Mildly Bearish"
    elif value >= 20:
        return "Bearish"
    else:
        return "Extremely Bearish"


def generate_report(composite: float, indicator_values: Dict[str, float], 
                   data_manager: DataManager) -> str:
    
    report = [
        "# Market Sentiment Analysis Report",
        f"**Date: {datetime.date.today().strftime('%B %d, %Y')}**",
        "",
        f"## Overall Sentiment: {get_sentiment_rating(composite)} ({composite:.1f}/100)",
        "",
        "### Individual Indicators:"
    ]
    
    sorted_indicators = sorted(indicator_values.items(), key=lambda x: x[1], reverse=True)
    
    for name, value in sorted_indicators:
        report.append(f"- **{name}**: {value:.1f}/100 - {get_sentiment_rating(value)}")
    
    report.append("")
    report.append("### Market Interpretation")
    
    # Add interpretation based on sentiment
    if composite >= 65:
        report.append("The market sentiment is currently bullish, suggesting positive momentum and favorable market conditions. This may indicate a good environment for growth-oriented investments, but be mindful of potentially stretched valuations.")
    elif composite >= 45:
        report.append("The market sentiment is neutral to mildly positive. The market appears to be in a balanced state with no strong directional bias. This suggests a selective approach to investments may be appropriate.")
    else:
        report.append("The market sentiment is currently bearish, suggesting caution is warranted. Risk management and defensive positioning may be appropriate until sentiment improves.")
    
    report.append("")
    report.append("### Key Observations")
    
    strongest = sorted_indicators[0]
    weakest = sorted_indicators[-1]
    
    report.append(f"- The strongest indicator is **{strongest[0]}** at {strongest[1]:.1f}/100, suggesting {get_observation(strongest[0], strongest[1])}.")
    report.append(f"- The weakest indicator is **{weakest[0]}** at {weakest[1]:.1f}/100, suggesting {get_observation(weakest[0], weakest[1])}.")
    
    if "Momentum" in indicator_values and "MarketBreadth" in indicator_values:
        mom = indicator_values["Momentum"]
        breadth = indicator_values["MarketBreadth"]
        
        if abs(mom - breadth) > 30:
            report.append(f"- There is a significant divergence between Momentum ({mom:.1f}) and Market Breadth ({breadth:.1f}), which could signal an upcoming shift in market direction.")
    
    report.append("")
    report.append("### Disclaimer")
    report.append("This analysis is for informational purposes only and should not be considered investment advice. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.")
    
    return "\n".join(report)


def get_observation(indicator: str, value: float) -> str:
    
    if indicator == "Momentum":
        if value >= 70:
            return "strong positive price momentum in the market"
        elif value <= 30:
            return "weak price momentum that may signal a downtrend"
        else:
            return "neutral price momentum"
            
    elif indicator == "NewHighsLows":
        if value >= 70:
            return "many stocks are reaching new highs, a positive breadth signal"
        elif value <= 30:
            return "few stocks are making new highs, indicating narrowing market participation"
        else:
            return "a balanced mix of stocks making new highs and lows"
            
    elif indicator == "MarketBreadth":
        if value >= 70:
            return "broad market participation in the current trend"
        elif value <= 30:
            return "limited market participation, which may indicate a weakening trend"
        else:
            return "moderate market breadth"
            
    elif indicator == "PutCall":
        if value >= 70:
            return "low put/call ratio indicating market optimism (potentially contrarian bearish)"
        elif value <= 30:
            return "high put/call ratio indicating market fear (potentially contrarian bullish)"
        else:
            return "balanced options sentiment"
            
    elif indicator == "JunkBond":
        if value >= 70:
            return "low credit spreads indicating comfort with risk"
        elif value <= 30:
            return "high credit spreads indicating credit stress"
        else:
            return "neutral credit conditions"
            
    elif indicator == "VIX":
        if value >= 70:
            return "low market volatility and complacency"
        elif value <= 30:
            return "high market volatility and fear"
        else:
            return "moderate market volatility"
            
    elif indicator == "SafeHaven":
        if value >= 70:
            return "funds flowing to risk assets rather than safe havens"
        elif value <= 30:
            return "funds flowing to safe havens rather than risk assets"
        else:
            return "balanced fund flows between risk assets and safe havens"
            
    elif indicator == "Economic":
        if value >= 70:
            return "strong economic indicators supporting market sentiment"
        elif value <= 30:
            return "weak economic indicators that may pressure markets"
        else:
            return "mixed economic signals"
            
    else:
        return "notable market conditions"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute composite market sentiment.")
    parser.add_argument("--fred_key", type=str, default="054d79a6dffb592fd462713e98e04d85", help="FRED API key")
    parser.add_argument("--start_date", type=str, default=None, help="Start date in YYYY-MM-DD format (default: 3 years ago)")
    parser.add_argument("--end_date", type=str, default=None, help="End date in YYYY-MM-DD format (default: yesterday)")
    parser.add_argument("--output", type=str, default=None, help="Output file for report (default: None)")
    parser.add_argument("--weights", type=str, default=None, help="JSON file with indicator weights")
    parser.add_argument("--no_chart", action="store_true", help="Skip generating charts")
    args = parser.parse_args()
    
    today = datetime.date.today()
    default_end = today - datetime.timedelta(days=1)
    default_start = default_end - datetime.timedelta(days=3 * 365)
    start_date = args.start_date if args.start_date else str(default_start)
    end_date = args.end_date if args.end_date else str(default_end)
    
    logger.info("Using date range from %s to %s", start_date, end_date)
    
    weights = None
    if args.weights:
        try:
            with open(args.weights, 'r') as f:
                weights = json.load(f)
            logger.info(f"Loaded custom weights: {weights}")
        except Exception as e:
            logger.error(f"Error loading weights file: {e}")
    
    logger.info("Initializing data manager")
    dm = DataManager(start_date, end_date, args.fred_key)
    
    logger.info("Computing composite sentiment")
    composite = CompositeSentiment(dm, weights)
    comp_value, indicator_values, historical_values = composite.compute()
    
    indicator_names = list(indicator_values.keys())
    
    print("\n" + "="*60)
    print(f"  MARKET SENTIMENT ANALYSIS: {today.strftime('%B %d, %Y')}")
    print("="*60)
    
    for name in indicator_names:
        val = indicator_values[name]
        print(f"{name:15s}: {val:.2f}/100 - {get_sentiment_rating(val)}")
    
    print("-"*60)
    print(f"COMPOSITE SENTIMENT: {comp_value:.2f}/100 - {get_sentiment_rating(comp_value)}")
    print("="*60 + "\n")
    
    if not args.no_chart:
        logger.info("Generating visualizations")
        try:
            gauge_fig = gauge_plot(comp_value)
            gauge_fig.show()
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
    
    if args.output:
        logger.info(f"Generating report to {args.output}")
        report = generate_report(comp_value, indicator_values, dm)
        
        with open(args.output, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
