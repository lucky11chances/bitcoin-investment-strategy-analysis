# Quantitative Strategy Technical Summary

## Strategy Overview

This quantitative strategy is a multi-factor Bitcoin trading system based on technical analysis indicators and market structure features to construct dynamic position signals. The goal is to maximize the Sharpe ratio on the training set.

---

## I. Strategy Formation Process

### 1. Factor Design Phase
The strategy first identified 10 key factors that may influence Bitcoin price trends, covering four dimensions: momentum, trend, volatility, and market cycle.

### 2. Factor Standardization
Uses **Rolling Z-score Standardization** (90-day window) to ensure factors of different dimensions are comparable:
```
Z-score = (Current Value - Rolling Mean) / Rolling Std
```

### 3. Signal Synthesis
Synthesizes standard factors into a composite signal via weighted sum:
```
Score_t = w₁×F₁_t + w₂×F₂_t + ... + w₁₀×F₁₀_t
```

### 4. Position Mapping
Uses **Sigmoid Function** to convert signal into 0-100% continuous position ratio:
```
Position_t = 1 / (1 + e^(-Score_t))
```
- Score > 2 → Position approaches 100% (Strong Bull)
- Score ≈ 0 → Position approx 50% (Neutral)
- Score < -2 → Position approaches 0% (Strong Bear)

### 5. Weight Optimization
Adopts **Random Walk Search Algorithm**, finding optimal weight combination through 300 iterations, with L2 regularization to prevent overfitting:
```
Objective Function = Sharpe Ratio - 0.012 × ||weights||²
Step Size: 0.05 (Conservative Search)
Transaction Cost: 15 bps (Real Constraint)
```

---

## II. Ten Technical Factors

### 📈 Momentum Factors
| Factor | Calculation | Economic Meaning |
|--------|-------------|------------------|
| **MOM_20** | 20-day Return | Short-term price trend strength |
| **MOM_60** | 60-day Return | Medium-term price trend strength |

**Logic**: Price momentum persistence, "Winners keep winning" effect.

---

### 📊 Moving Average Spread Factors
| Factor | Calculation | Economic Meaning |
|--------|-------------|------------------|
| **MA_50_SPREAD** | (Price - MA50) / MA50 | Price deviation relative to medium-term MA |
| **MA_200_SPREAD** | (Price - MA200) / MA200 | Price deviation relative to long-term MA |

**Logic**: 
- Price > MA → Bull Market
- Price < MA → Bear Market or Overbought/Oversold signal

---

### 📉 Volatility Factors
| Factor | Calculation | Economic Meaning |
|--------|-------------|------------------|
| **VOL_20** | 20-day Return Std | Short-term market volatility degree |
| **ATR_PCT_14** | 14-day ATR / Price | True fluctuation range (considering gaps) |
| **VOL_RATIO_20** | Current Volume / 20-day Avg Vol | Volume relative activity |

**Logic**: 
- High volatility may indicate trend reversal or breakout.
- Volume expansion usually accompanies important market moves.

---

### 🎯 Price Position Factors
| Factor | Calculation | Economic Meaning |
|--------|-------------|------------------|
| **PRICE_POS_60** | (Price - 60d Low) / (60d High - 60d Low) | Price relative position in 60-day range |
| **CLOSE_POS** | (Close - Low) / (High - Low) | Daily price position within intraday range |

**Logic**: 
- Near range top → Overbought, possible pullback
- Near range bottom → Oversold, possible rebound

---

### 🔄 Market Cycle Factor
| Factor | Calculation | Economic Meaning |
|--------|-------------|------------------|
| **POST_HALVING** | Days since last halving / 1460 | Bitcoin halving cycle position |

**Logic**: 
- Bitcoin halves every 4 years, historically bull markets occur after halving.
- Cyclical patterns may influence price trends.

---

## III. Weight Allocation Mechanism

### Random Walk Search Algorithm

#### Algorithm Flow
```
1. Initialize: Randomly generate 10 weights w = [w₁, w₂, ..., w₁₀]
2. Loop 2000 steps:
   a. Generate perturbation: Δw ~ N(0, 0.1²)  # Normal distribution random noise
   b. Candidate weights: w_new = w_old + Δw
   c. Calculate Sharpe: Backtest w_new performance on training set
   d. Greedy Update: if Sharpe(w_new) > Sharpe(w_old):
                  w_old = w_new  # Accept better solution
   e. Record Best: Track historical best Sharpe and corresponding weights
3. Output: Return weight combination with highest Sharpe in 2000 steps
```

#### Key Parameters
- **Steps**: 2000 steps (Balance calculation cost and optimization effect)
- **Step Size**: Std dev 0.1 (Control exploration range)
- **Objective Function**: Sharpe Ratio (Risk-adjusted return)
- **Risk-Free Rate**: 3% Annualized

#### Optimization Result
Final Weights: `[1.70, -6.83, -1.58, 4.51, -0.69, 3.13, 2.07, 7.83, 3.29, -2.19]`

Training Set Best Sharpe: **2.21**

---

## IV. Weight Interpretation

### Positive Weight Factors (Bull Signal)
| Factor | Weight | Explanation |
|--------|--------|-------------|
| **PRICE_POS_60** | +7.83 | Increase position when price is high |
| **MA_50_SPREAD** | +4.51 | Bullish when price above 50-day MA |
| **CLOSE_POS** | +3.29 | Add position when intraday close is strong |
| **ATR_PCT_14** | +3.13 | Moderately add position when volatility rises |
| **VOL_RATIO_20** | +2.07 | Follow trend when volume expands |
| **MOM_20** | +1.70 | Hold when short-term momentum is positive |

### Negative Weight Factors (Bear Signal)
| Factor | Weight | Explanation |
|--------|--------|-------------|
| **MOM_60** | -6.83 | Reduce position when medium-term gain is too large (Mean Reversion) |
| **POST_HALVING** | -2.19 | Lower position in later halving cycle |
| **MA_200_SPREAD** | -1.58 | Cautious when far from long-term MA |
| **VOL_20** | -0.69 | Lower risk exposure when short-term volatility is too high |

### Strategy Characteristics
1. **Trend Following + Mean Reversion Hybrid**: Long on short-term momentum, reverse on excessive medium-term gains.
2. **High Weight Concentration**: PRICE_POS_60 and MOM_60 dominate decision making.
3. **Risk Management**: Volatility factors regulate position intensity.

---

## V. Strategy Performance & Overfitting Analysis

### 📊 Training Set Performance (2010-2020)
- **Final Value**: $2.66B
- **Sharpe Ratio**: 1.98
- **Performance**: Slightly better than DCA ($2.31B), but far below HODL ($5.3B)

### 📉 Test Set Performance (2023-2024)
- **Final Value**: $18,156
- **Sharpe Ratio**: 1.08
- **Performance**: Lost to HODL and DCA, proving insufficient strategy generalization ability.

### ⚠️ Overfitting Analysis

**Reasons for Performance Decline**:
1. **Weights Optimized for Historical Environment**: 2010-2020 market features differ from 2023-2024.
2. **Optimization Parameter Selection**: Despite regularization, some overfitting remains.
3. **Factor Timeliness**: Some technical factors fail in new market environment.

**Anti-Overfitting Measures (Implemented)**:
1. ✅ **L2 Regularization**: Penalty coefficient 0.012, limits excessive weights.
2. ✅ **Transaction Cost**: 15 bps real cost constraint.
3. ✅ **Conservative Optimization**: 300 steps (vs original 2000 steps).
4. ✅ **Small Step Search**: Step size 0.05 (vs original 0.5).
5. ✅ **Dynamic Position**: Sigmoid outputs continuous position, avoiding extreme switching.

**Improvement Directions**:
1. **Time Series Cross Validation**: Rolling window to verify strategy stability.
2. **Factor Importance Analysis**: Remove unstable factors.
3. **Ensemble Learning**: Combine multiple models to reduce single model risk.
4. **Online Learning**: Periodically update weights to adapt to market changes.
5. **Stop Loss Mechanism**: Add max drawdown limit.

---

## VI. Technical Implementation Highlights

### 1. Dynamic Position Management
```python
# Sigmoid function supports 0-100% continuous position
Position_t = 1.0 / (1 + exp(-Score_t))
```
- Not just full position/empty position.
- Supports 30%, 60%, 85% any position.
- Flexibly adjust based on signal strength.

### 2. L2 Regularization Optimization
```python
# Penalize extreme weights
Penalized_Sharpe = Raw_Sharpe - 0.012 × ||weights||²
```
- Prevents excessive weights.
- Improves model generalization ability.
- Reduces overfitting risk.

### 3. Real Cost Simulation
- Transaction Cost: 15 bps
- Calculate cost for every position change.
- Closer to actual trading environment.

---

## VII. Summary

This quantitative strategy demonstrates the application framework of **Multi-Factor Models** in the cryptocurrency field, capturing market dynamics through technical factors and automatically searching for optimal weights using random optimization algorithms.

**Core Value**:
- ✅ Complete quantitative strategy development process.
- ✅ Demonstrates overfitting issues and preventive measures.
- ✅ Real backtesting framework (including costs).
- ✅ Dynamic position management mechanism.

**Key Lesson**:
> **Excellent training set performance ≠ Actual profitability. Out-of-sample validation is the only standard for strategy reliability.**

For actual trading, suggested:
1. Strict out-of-sample testing.
2. Multi-period cross-validation.
3. Conservative parameter selection.
4. Comprehensive risk management.

---

**Strategy File**: `src/strategies/quant_rf.py`
**Tech Stack**: Python + Pandas + NumPy
**Data Period**: Daily Frequency Data
**Author**: Haoyu
**Update Time**: December 14, 2025
