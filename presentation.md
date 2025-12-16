# Bitcoin Investment Strategies Analysis
## DCA vs HODL vs Quantitative

---

# Agenda

1.  **Project Overview**
2.  **Dataset Breakdown**
3.  **Strategy Assumptions (DCA & HODL)**
4.  **Quant Strategy Development**
5.  **Performance Comparison**
6.  **Key Conclusions**

---

# 1. Project Overview

*   **Goal**: Compare three classic Bitcoin investment strategies from 2010 to 2024.
*   **Strategies**:
    *   **HODL**: Lump-sum investment, buy and hold.
    *   **DCA (Dollar-Cost Averaging)**: Fixed monthly investment.
    *   **Quantitative**: Multi-factor technical analysis model.
*   **Key Finding**: Simple strategies (DCA/HODL) often outperform complex over-optimized models in out-of-sample tests.

---

# 2. Dataset Breakdown

The dataset is splits into three periods to ensure rigorous training and testing.

## Data Columns
*   `Start`, `End` (Date ranges)
*   `Open`, `High`, `Low`, `Close` (Price data)
*   `Volume`, `Market Cap`

## Data Splits (Counts)
| Split | Years | Rows (Approx) | Purpose |
| :--- | :--- | :--- | :--- |
| **Training** | 2010 - 2020 | ~3,806 | Strategy development & Weight optimization |
| **Validation** | 2021 - 2022 | ~730 | Hyperparameter tuning (not fully used in final report) |
| **Testing** | 2023 - 2024 | ~545 | Out-of-sample performance verification |
| **Total** | | ~5,081 | |

---

# 3. Strategy Assumptions

## HODL (Buy and Hold)
*   **"Time in the market beats timing the market."**
*   **Assumption 1**: Lump-sum investing at the start is optimal.
*   **Assumption 2**: Short-term volatility is noise; long-term trend is up.
*   **Assumption 3**: Investor has high risk tolerance (ignores >90% drawdowns).
*   **Implementation**: Buy \$12,500 on Day 1, hold forever.

## DCA (Dollar-Cost Averaging)
*   **"Smooth out the volatility."**
*   **Assumption 1**: Timing is impossible; consistency is key.
*   **Assumption 2**: Buying fixed amounts automatically buys more when cheap and less when expensive.
*   **Assumption 3**: Emotional stability is worth the potential lower absolute return compared to perfect HODL.
*   **Implementation**: Invest \$1,000 monthly for 13 months (Total \$13,000).

---

# 4. Quant Strategy Formation (1/2)

A custom **Random Forest / Linear Combination** style strategy based on 10 technical factors.

## The Process
1.  **Factor Engineering**: Created 10 factors covering Momentum, Trend, Volatility, and Cycle.
2.  **Standardization**: Rolling Z-score (90-day window) to normalize inputs.
3.  **Signal Synthesis**: Weighted sum of factors.
4.  **Position Mapping**: `Sigmoid` function converts score to 0-100% position (Dynamic sizing).
5.  **Optimization**: **Random Walk Search** (300 steps) to find weights that maximize Sharpe Ratio on Training Data.

---

# 4. Quant Strategy Formation (2/2)

## Key Technical Factors (10 Total)
*   **Price Position**: `PRICE_POS_60` (High weight: +2.56) - *Mean Reversion*
*   **Momentum**: `MOM_60` (Weight: -1.87) - *Counter-trend*
*   **Volatility**: `ATR_PCT_14` (Weight: +1.32) - *Volatility adjustment*
*   **Market Cycle**: `POST_HALVING` - *Halving cycle awareness*
*   *And others: MA Spreads, RSI, Volume Ratios...*

## Constraints
*   **Transaction Cost**: 15 bps (0.15%) per trade.
*   **Regularization**: L2 penalty to prevent extreme weights.

---

# 5. Performance Comparison

## Training Set (2010-2020) - *In-Sample*
*   **Quant**: Sharpe **1.98**, Final Value \$2.66B (Over-optimized!)
*   **DCA**: Sharpe 1.78
*   **HODL**: Sharpe 1.70

## Testing Set (2023-2024) - *Out-of-Sample*
*   **HODL**: Sharpe **3.26** (Best Return: +268%)
*   **DCA**: Sharpe **3.04** (Best Risk-Adjusted: +141%, Low Drawdown)
*   **Quant**: Sharpe **-5.14** (Loss: -25%)

---

# 6. Conclusion & Insights

*   **Overfitting is Real**: The Quant strategy learned historical noise (2010-2020) that did not repeat in 2023-2024.
*   **Complexity Cost**: High turnover (trading every ~1.2 days) accumulated significant transaction costs.
*   **DCA Wins on Reliability**:
    *   No stress about timing.
    *   Excellent risk-adjusted returns (Sharpe 3.04).
    *   Lowest maximum drawdown in testing (-20% vs others).
*   **Recommendation**:
    *   **General Investors**: Stick to **DCA**.
    *   **Quant Devs**: Strict out-of-sample testing is non-negotiable.

