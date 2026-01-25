import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import TEST_DATA_PATH, TRAINED_WEIGHTS, INITIAL_CAPITAL
from utils import load_bitcoin_data

# =========================
# 1. Data Loading and Preprocessing
# =========================

def load_btc_data(csv_path) -> pd.DataFrame:
    """
    Read BTC daily data, requiring at least:
    Date/Start, Open, High, Low, Close, Volume
    """
    df = load_bitcoin_data(csv_path)
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input CSV.")

    # Daily Return (log return)
    df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
    return df


# =========================
# 2. Factor Calculation (10 Factors)
# =========================

def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 factor columns to the original df and return the df with factors.
    """
    eps = 1e-9

    # ----- 1) Momentum and Moving Average -----

    # Moving Average
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # MOM_20, MOM_60
    df['MOM_20'] = df['Close'] / df['Close'].shift(20) - 1.0
    df['MOM_60'] = df['Close'] / df['Close'].shift(60) - 1.0

    # MA_50_SPREAD, MA_200_SPREAD
    df['MA_50_SPREAD'] = (df['Close'] - df['MA_50']) / (df['MA_50'] + eps)
    df['MA_200_SPREAD'] = (df['Close'] - df['MA_200']) / (df['MA_200'] + eps)

    # ----- 2) Volatility / ATR -----

    # log return already in df['ret']

    # VOL_20 (realized vol, based on log return)
    df['VOL_20'] = df['ret'].rolling(window=20).std()

    # True Range
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = np.maximum.reduce([tr1, tr2, tr3])

    # ATR_14
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df['ATR_PCT_14'] = df['ATR_14'] / (df['Close'] + eps)

    # ----- 3) Volume Factors -----

    df['VOL_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['VOL_RATIO_20'] = df['Volume'] / (df['VOL_MA_20'] + eps)

    # ----- 4) Price Position / Candle Structure -----

    # High & Low of past 60 days
    df['HIGH_60'] = df['High'].rolling(window=60).max()
    df['LOW_60'] = df['Low'].rolling(window=60).min()
    df['PRICE_POS_60'] = (df['Close'] - df['LOW_60']) / (df['HIGH_60'] - df['LOW_60'] + eps)

    # CLOSE_POS: Close position in High-Low range of the day
    df['CLOSE_POS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + eps)

    # ----- 5) Halving Cycle Factor -----

    # Known Halving Dates (UTC approx)
    halving_dates = [
        pd.Timestamp('2012-11-28'),
        pd.Timestamp('2016-07-09'),
        pd.Timestamp('2020-05-11'),
    ]

    # Find the nearest (past) halving date for each day
    last_halving = []
    for d in df['Date']:
        past_halvings = [h for h in halving_dates if h <= d]
        if len(past_halvings) == 0:
            last_halving.append(pd.NaT)
        else:
            last_halving.append(max(past_halvings))
    df['LAST_HALVING'] = last_halving

    # Days Difference
    df['DAYS_SINCE_HALVING'] = (df['Date'] - df['LAST_HALVING']).dt.days

    # POST_HALVING: 1 year after halving = 1, else 0 (including period before any halving)
    df['POST_HALVING'] = np.where(
        (df['DAYS_SINCE_HALVING'] >= 0) & (df['DAYS_SINCE_HALVING'] <= 365),
        1.0,
        0.0
    )

    # Delete intermediate helper columns when returning
    factor_cols = [
        'MOM_20',
        'MOM_60',
        'MA_50_SPREAD',
        'MA_200_SPREAD',
        'VOL_20',
        'ATR_PCT_14',
        'VOL_RATIO_20',
        'PRICE_POS_60',
        'CLOSE_POS',
        'POST_HALVING',
    ]

    return df, factor_cols


# =========================
# 3. Factor Standardization (Rolling Z-score, avoid future info)
# =========================

def rolling_standardize(df: pd.DataFrame, factor_cols, window: int = 252) -> pd.DataFrame:
    """
    Rolling Z-score for each factor:
    z_t = (x_t - mean_{t-1}) / std_{t-1}
    Use rolling().mean().shift(1) to avoid future data leakage.
    """
    for col in factor_cols:
        roll_mean = df[col].rolling(window=window).mean().shift(1)
        roll_std = df[col].rolling(window=window).std().shift(1)
        df[col + '_Z'] = (df[col] - roll_mean) / (roll_std + 1e-9)
    z_cols = [c + '_Z' for c in factor_cols]
    return df, z_cols


# =========================
# 4. Position Generation & Backtest
# =========================

def weights_to_positions(df: pd.DataFrame, z_cols, weights: np.ndarray) -> pd.Series:
    """
    Generate daily positions in [0, 1] range given a set of weights (length == len(z_cols)).
    Use weighted sum + Sigmoid mapping to [0,1].
    """
    if len(weights) != len(z_cols):
        raise ValueError("Length of weights must match number of factor columns.")

    X = df[z_cols].values
    # Linear Score
    scores = np.dot(X, weights)
    # Sigmoid Mapping to (0,1)
    positions = 1.0 / (1.0 + np.exp(-scores))
    return pd.Series(positions, index=df.index, name='position')


def backtest(df: pd.DataFrame, positions: pd.Series, tc_bps: float = 5.0):
    """
    Simple Backtest:
    - pos_t used for return from t to t+1 (using ret_{t+1})
    - Transaction Cost: |pos_t - pos_{t-1}| * tc (bps) for each position change
    """
    df = df.copy()
    df['position'] = positions

    # Actual holdings use previous day's signal (avoid future function)
    df['position_shifted'] = df['position'].shift(1).fillna(0.0)

    # Position Change (for cost)
    df['position_change'] = df['position_shifted'].diff().fillna(df['position_shifted'])

    # Transaction Cost (bps to decimal)
    tc = tc_bps / 10000.0
    df['cost'] = np.abs(df['position_change']) * tc

    # Strategy Daily Return
    df['strategy_ret'] = df['position_shifted'] * df['ret'] - df['cost']

    # Remove NaN from early undefined factors
    strat = df.dropna(subset=['strategy_ret'])

    if strat['strategy_ret'].std() == 0 or np.isnan(strat['strategy_ret'].std()):
        sharpe = 0.0
    else:
        avg_daily = strat['strategy_ret'].mean()
        std_daily = strat['strategy_ret'].std()
        sharpe = np.sqrt(252) * avg_daily / std_daily

    cum_return = np.exp(strat['strategy_ret'].cumsum()) - 1.0

    return {
        'sharpe': sharpe,
        'cum_return': cum_return.iloc[-1] if len(cum_return) > 0 else 0.0,
        'equity_curve': np.exp(strat['strategy_ret'].cumsum()),
        'df': strat
    }


# =========================
# 5. Main Program - Test Set (2023-2024)
# =========================

if __name__ == "__main__":
    print("=== Quant Strategy Test (2023-2024) ===")
    
    # Load Test Data
    df = load_btc_data(TEST_DATA_PATH)
    df, factor_cols = compute_factors(df)
    df, z_cols = rolling_standardize(df, factor_cols, window=252)
    
    # Drop early period where factors are completely NaN
    df = df.dropna(subset=z_cols + ['ret']).reset_index(drop=True)
    
    print(f"Test data: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
    print(f"Number of trading days: {len(df)}")
    
    # Backtest using trained weights
    positions = weights_to_positions(df, z_cols, TRAINED_WEIGHTS)
    res = backtest(df, positions, tc_bps=5.0)
    
    final_value = INITIAL_CAPITAL * (1.0 + res['cum_return'])
    
    print(f"\nSharpe Ratio: {res['sharpe']:.4f}")
    print(f"Cumulative Return: {res['cum_return']:.2%}")
    print(f"Final Value (USD): ${final_value:,.2f}")
