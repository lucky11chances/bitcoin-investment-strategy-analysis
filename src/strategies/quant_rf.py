import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    TRAIN_DATA_PATH,
    TRAINED_WEIGHTS,
    TRADING_COST_BPS,
    MOMENTUM_SHORT_WINDOW,
    MOMENTUM_LONG_WINDOW,
    MA_SHORT_WINDOW,
    MA_LONG_WINDOW,
    VOLATILITY_WINDOW,
    ATR_WINDOW,
    PRICE_POSITION_WINDOW,
    ROLLING_Z_WINDOW,
    OPTIMIZATION_STEPS,
    OPTIMIZATION_STEP_SIZE,
    RISK_FREE_RATE_ANNUAL,
    DAYS_PER_YEAR,
    INITIAL_CAPITAL
)
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
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input CSV.")

    # Daily Return (log return)
    df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
    return df


# =========================
# 2. Factor Calculation (10 Factors, all periods <= 90 days)
# =========================

def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 factor columns to the original df and return the df with factors.
    All rolling windows <= 90 days.
    """
    eps = 1e-9

    # ----- 1) Momentum and Moving Average -----

    # Moving Average
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_90'] = df['Close'].rolling(window=90).mean()   # CHANGED: MA_200 changed to MA_90

    # MOM_20, MOM_60 (both <= 90)
    df['MOM_20'] = df['Close'] / df['Close'].shift(20) - 1.0
    df['MOM_60'] = df['Close'] / df['Close'].shift(60) - 1.0

    # MA_50_SPREAD, MA_90_SPREAD (both <= 90)
    df['MA_50_SPREAD'] = (df['Close'] - df['MA_50']) / (df['MA_50'] + eps)
    df['MA_90_SPREAD'] = (df['Close'] - df['MA_90']) / (df['MA_90'] + eps)   # CHANGED: Use 90 days

    # ----- 2) Volatility / ATR -----

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

    # High & Low of past 60 days (<= 90)
    df['HIGH_60'] = df['High'].rolling(window=60).max()
    df['LOW_60'] = df['Low'].rolling(window=60).min()
    df['PRICE_POS_60'] = (df['Close'] - df['LOW_60']) / (df['HIGH_60'] - df['LOW_60'] + eps)

    # CLOSE_POS: Close position in High-Low range of the day
    df['CLOSE_POS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + eps)

    # ----- 5) Halving Cycle Factor (Shortened to 90 days post halving) -----

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

    # POST_HALVING: 1 year after halving = 1, else 0
    df['POST_HALVING'] = np.where(
        (df['DAYS_SINCE_HALVING'] >= 0) & (df['DAYS_SINCE_HALVING'] <= 90),   # CHANGED: 365 -> 90
        1.0,
        0.0
    )

    # Final 10 Factor Columns
    factor_cols = [
        'MOM_20',
        'MOM_60',
        'MA_50_SPREAD',
        'MA_90_SPREAD',   # CHANGED: Replacing original MA_200_SPREAD
        'VOL_20',
        'ATR_PCT_14',
        'VOL_RATIO_20',
        'PRICE_POS_60',
        'CLOSE_POS',
        'POST_HALVING',
    ]

    return df, factor_cols


# =========================
# 3. Factor Standardization (Rolling Z-score, window changed to <= 90)
# =========================

def rolling_standardize(df: pd.DataFrame, factor_cols, window: int = 90) -> pd.DataFrame:
    """
    Rolling Z-score for each factor:
    z_t = (x_t - mean_{t-1}) / std_{t-1}
    Use rolling().mean().shift(1) to avoid future data leakage.
    Window changed to 90 days default (originally 252).
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

def weights_to_positions(df: pd.DataFrame, z_cols, weights: np.ndarray, max_position: float = 1.0) -> pd.Series:
    """
    Generate daily positions in [0, max_position] range given a set of weights.
    Use weighted sum + Sigmoid mapping, supports arbitrary position allocation between 0-100%.
    
    Args:
        max_position: Max position ratio, default 1.0 (100%), allows full position.
    
    Note:
        Sigmoid function outputs continuous position values, not just 0% or 100%,
        e.g., 30% BTC + 70% Cash, or 60% BTC + 40% Cash.
    """
    if len(weights) != len(z_cols):
        raise ValueError("Length of weights must match number of factor columns.")

    X = df[z_cols].values
    # Linear Score
    scores = np.dot(X, weights)
    # Sigmoid Mapping to (0,1), then scale to [0, max_position]
    positions = max_position / (1.0 + np.exp(-scores))
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
# 5. Random Walk Weight Search
# =========================

def random_walk_search(df: pd.DataFrame,
                       z_cols,
                       n_steps: int = 300,
                       step_size: float = 0.05,
                       tc_bps: float = 15.0,
                       l2_penalty: float = 0.012,
                       seed: int = 42):
    """
    Simple "Random Walk + Hill Climbing", adding regularization to prevent overfitting:
    - Initialize random weights
    - Add Gaussian noise to weights at each step
    - If new weights have higher Sharpe (with L2 penalty), accept, else reject
    - Return best weights and performance
    
    Improvement (Target: Train set $2.5B-$3B, slightly better than DCA; Test set performance drops):
    - Optimization Steps: 350 (Moderate optimization)
    - Step Size: 0.06 (Smaller step, conservative search)
    - Transaction Cost: 12 bps
    - L2 Regularization: 0.01 (Stronger penalty, limit weight size)
    - Max Position: 100% (Allow full position, but control actual position via small weights)
    
    Strategy Features:
    - Supports continuous position adjustment (arbitrary value between 0%-100%)
    - Use small weights to make sigmoid output smoother, avoiding extreme full/empty positions
    """
    rng = np.random.default_rng(seed)
    n_factors = len(z_cols)

    # Initialize weights (small random numbers for smoother sigmoid output)
    w = rng.normal(loc=0.0, scale=0.15, size=n_factors)

    # Initial Evaluation
    pos = weights_to_positions(df, z_cols, w)
    res = backtest(df, pos, tc_bps=tc_bps)
    # Add L2 Regularization Penalty
    best_sharpe = res['sharpe'] - l2_penalty * np.sum(w ** 2)
    best_w = w.copy()
    best_raw_sharpe = res['sharpe']

    print(f"[Init] Raw Sharpe = {best_raw_sharpe:.4f}, Penalized = {best_sharpe:.4f}")

    for step in range(1, n_steps + 1):
        # Proposal: Add noise to current weights
        proposal = best_w + rng.normal(loc=0.0, scale=step_size, size=n_factors)

        pos_p = weights_to_positions(df, z_cols, proposal)
        res_p = backtest(df, pos_p, tc_bps=tc_bps)
        # Calculate Penalized Sharpe
        sharpe_p = res_p['sharpe'] - l2_penalty * np.sum(proposal ** 2)

        # Accept if better
        if sharpe_p > best_sharpe:
            best_sharpe = sharpe_p
            best_w = proposal
            best_raw_sharpe = res_p['sharpe']
            if step % 50 == 0 or step < 10:
                print(f"[Step {step}] Raw Sharpe = {best_raw_sharpe:.4f}, Penalized = {best_sharpe:.4f}")

    print(f"\n[Final] Best Raw Sharpe = {best_raw_sharpe:.4f}, Weight L2 Norm = {np.linalg.norm(best_w):.4f}")
    return best_w, best_raw_sharpe


# =========================
# 6. Main Program Example
# =========================

if __name__ == "__main__":
    # BTC Training Data Path
    from config import TRAIN_DATA_PATH
    
    df = load_btc_data(TRAIN_DATA_PATH)
    df, factor_cols = compute_factors(df)

    # Standardize window changed to <= 90 (using 90 here)
    df, z_cols = rolling_standardize(df, factor_cols, window=90)  # CHANGED: 252 -> 90

    # Drop early period where factors are completely NaN
    df = df.dropna(subset=z_cols + ['ret']).reset_index(drop=True)

    print("Factor columns (Z-scored):", z_cols)

    # ==== 6.1 Manually set a group of weights, quick look at Sharpe ====
    # Note: Factor order unchanged, just MA_200_SPREAD replaced by MA_90_SPREAD
    manual_weights = np.array([0.8, 1.0, 0.5, 0.5, -0.3, -0.3, 0.2, 0.4, 0.3, 0.5])
    positions = weights_to_positions(df, z_cols, manual_weights)
    res_manual = backtest(df, positions, tc_bps=5.0)
    print(f"Manual weights Sharpe: {res_manual['sharpe']:.4f}, "
          f"CumReturn: {res_manual['cum_return']:.2f}")

    # ==== 6.2 Random Walk Search for weights ====
    best_w, best_sharpe = random_walk_search(
        df,
        z_cols,
        n_steps=2000,
        step_size=0.2,
        tc_bps=5.0,
        seed=42
    )
    print(f"\nBest weights after search: {best_w}")
    print(f"Best Sharpe: {best_sharpe:.4f}")

    # ==== 6.3 Final Backtest and Output Final Value (USD) ====
    final_pos = weights_to_positions(df, z_cols, best_w)
    final_res = backtest(df, final_pos, tc_bps=5.0)
    final_value = INITIAL_CAPITAL * (1.0 + final_res['cum_return'])
    print(f"\n=== Final Results ===")
    print(f"Sharpe Ratio: {final_res['sharpe']:.4f}")
    print(f"Cumulative Return: {final_res['cum_return']:.2%}")
    print(f"Final Value (USD): ${final_value:,.2f}")


# =========================
# 7. Run function for main.py
# =========================

def run(retrain: bool = False) -> dict:
    """
    Run the quantitative strategy and return metrics
    
    Args:
        retrain: If True, re-optimize weights (takes ~2 minutes). 
                 If False, use pre-trained weights (fast).
    
    Returns dict with: sharpe_ratio, cum_return, final_value, best_weights
    """
    df = load_btc_data(TRAIN_DATA_PATH)
    df, factor_cols = compute_factors(df)
    df, z_cols = rolling_standardize(df, factor_cols, window=90)
    df = df.dropna(subset=z_cols + ['ret']).reset_index(drop=True)

    # Use pre-trained weights or re-optimize
    if retrain:
        print("⚠️  Re-training weights (this may take a few minutes)...")
        best_w, best_sharpe = random_walk_search(
            df,
            z_cols,
            n_steps=2000,
            step_size=0.2,
            tc_bps=5.0,
            seed=42
        )
        print(f"✓ Training complete. New Sharpe: {best_sharpe:.4f}")
    else:
        best_w = TRAINED_WEIGHTS

    # Final backtest
    final_pos = weights_to_positions(df, z_cols, best_w)
    final_res = backtest(df, final_pos, tc_bps=5.0)
    final_value = INITIAL_CAPITAL * (1.0 + final_res['cum_return'])

    # Convert weights to list if it's a numpy array
    weights_list = best_w.tolist() if hasattr(best_w, 'tolist') else list(best_w)

    return {
        'sharpe_ratio': final_res['sharpe'],
        'sortino_ratio': None,  # Not calculated in this version
        'max_drawdown': None,   # Not calculated in this version
        'cum_return': final_res['cum_return'],
        'final_value': final_value,
        'best_weights': weights_list
    }
