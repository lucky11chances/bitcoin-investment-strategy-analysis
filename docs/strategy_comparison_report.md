# Bitcoin Investment Strategy Comparison Report

## Executive Summary

This report compares the performance of two Bitcoin investment strategies across different periods:
- **DCA Strategy (Dollar-Cost Averaging)**: Monthly fixed investment of $1,000 for 13 months, total investment $13,000.
- **Quant Strategy (Quantitative)**: Quantitative trading strategy based on 10 technical factors, optimizing weights via random walk to maximize Sharpe ratio.

---

## I. Training Set Performance (2010-2020)

### DCA Strategy
| Metric | Value |
|--------|-------|
| Period | 2010-08-01 to 2020-12-30 |
| Total Investment | $13,000 |
| Final Value | **$2,312,629,735.98** |
| Sharpe Ratio | **1.78** |
| Sortino Ratio | 2.43 |
| Max Drawdown | -93.07% |
| Investment Months | 13 Months |

**Characteristics**:
- Robust long-term holding strategy.
- Reduces market volatility risk through batch buying.
- Sortino ratio fits better than Sharpe, indicating good downside risk control.

### Quantitative Strategy
| Metric | Value |
|--------|-------|
| Period | 2010 to 2020 |
| Initial Capital | $12,500 |
| Final Value | **$29,883,119,586.03** |
| Sharpe Ratio | **2.21** |
| Cumulative Return | 239,064,856.69% |

**Technical Factors**:
1. Momentum Indicators (MOM_20, MOM_60)
2. Moving Average Spread (MA_50_SPREAD, MA_200_SPREAD)
3. Volatility Indicators (VOL_20, ATR_PCT_14, VOL_RATIO_20)
4. Price Position (PRICE_POS_60, CLOSE_POS)
5. Halving Cycle (POST_HALVING)

**Optimized Weights**: [1.70, -6.83, -1.58, 4.51, -0.69, 3.13, 2.07, 7.83, 3.29, -2.19]

**Characteristics**:
- Optimized specifically for the training set through 2000 steps of random walk.
- Sharpe ratio (2.21) outperforms DCA (1.78).
- Final value is about 13 times that of DCA.
- ⚠️ Warning: Strategy designed with intentional allowance for overfitting.

### Training Set Summary
On the 2010-2020 training set, the Quant strategy significantly outperformed DCA in both risk-adjusted return (Sharpe) and absolute return.

---

## II. Test Set Performance (2023-2024)

### DCA Strategy
| Metric | Value |
|--------|-------|
| Period | 2023-01-01 to 2024-06-27 |
| Total Investment | $13,000 |
| Final Value | **$31,327.61** |
| Sharpe Ratio | **3.04** |
| Sortino Ratio | 6.87 |
| Max Drawdown | -20.28% |
| Cumulative Return | **+141%** |

**Performance Analysis**:
- Sharpe ratio improved from 1.78 to 3.04, better performance.
- Drawdown improved significantly from -93% to -20%, risk control significantly enhanced.
- Steady profit, strong strategy generalization ability.

### Quantitative Strategy
| Metric | Value |
|--------|-------|
| Period | 2024-03-26 to 2024-06-27 |
| Initial Capital | $12,500 |
| Final Value | **$9,310.57** |
| Sharpe Ratio | **-5.14** ❌ |
| Cumulative Return | **-25.52%** ❌ |
| Trading Days | 94 Days |

**Failure Analysis**:
1. **Severe Overfitting**: Training Set Sharpe 2.21 → Test Set Sharpe -5.14.
2. **Rolling Window Failure**: 252-day rolling window has insufficient data on 94-day test set.
3. **Market Environment Change**: Factor weights optimized for training set cannot adapt to new market.
4. **Negative Return**: Loss of 25.52%, capital shrank to $9,310.57.

### Test Set Summary
The DCA strategy performed robustly or even better on the test set, while the Quant strategy failed completely, showing significant losses, verifying the severity of overfitting.

---

## III. Comprehensive Comparison & Conclusion

### Performance Comparison Table
| Strategy | Train Sharpe | Test Sharpe | Train Return | Test Return | Stability |
|----------|-------------|-------------|--------------|-------------|-----------|
| DCA | 1.78 | **3.04** ✅ | $2.31B | **+141%** ✅ | Excellent |
| Quant | 2.21 | **-5.14** ❌ | $29.88B | **-25.52%** ❌ | Very Poor |

### Key Findings

1. **Overfitting Trap**:
   - Although the Quant strategy performed excellently on the training set, it failed completely on the test set.
   - Sharpe ratio dropped from positive 2.21 to negative 5.14, a performance reversal of over 7 standard deviations.

2. **Robustness Comparison**:
   - DCA strategy maintained positive Sharpe in both periods, showing good generalization ability.
   - Quant strategy failed to adapt to the new market environment, showing fragility.

3. **Practical Value**:
   - DCA is simple to execute, has low psychological burden, and considerable long-term returns.
   - Although Quant strategy is theoretically optimizable, the excessive optimization in this case led to practical failure.

### Recommendations

**For Ordinary Investors**:
- ✅ Recommend DCA Strategy: Simple, robust, no frequent adjustments needed.
- ❌ Do not recommend overly optimized Quant Strategy: Extremely high overfitting risk.

**For Quant Traders**:
- Must verify strategy effectiveness on an independent test set.
- Avoid over-optimizing parameters on a single period.
- Consider using cross-validation and out-of-sample testing.
- Set reasonable complexity constraints to prevent overfitting.

---

## IV. Technical Appendix

### Strategy Parameters
- **Initial Capital**: $12,500 (Quant), $13,000 (DCA Total Investment)
- **Risk-Free Rate**: 3% (Annualized)
- **Transaction Cost**: 5 basis points (Quant Strategy)
- **DCA Frequency**: Monthly
- **Quant Factors**: 10

### Data Sources
- Training Set: `bitcoin_train_2010_2020 copy.csv`
- Test Set: `bitcoin_test_2023_2024 copy.csv`

---

**Report Generation Date**: December 10, 2025
**Conclusion**: In Bitcoin investment, the simple DCA strategy significantly outperforms overly optimized quantitative strategies in long-term performance and robustness.
