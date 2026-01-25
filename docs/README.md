# Bitcoin Investment Strategies: HODL vs DCA vs Quantitative

Comparative analysis project implementing and comparing the performance of three classic investment strategies from 2010 to 2024.

## 📊 Strategy Overview

### 1. HODL Strategy (Buy and Hold)
One-time investment of $13,000, held long-term without operation.

### 2. DCA Strategy (Dollar-Cost Averaging)
Monthly fixed investment of $1,000 for 13 months (Total investment $13,000).

### 3. Quantitative Strategy
Quantitative trading system based on 10 technical factors, optimizing weights via random walk.

## 📈 Core Results

### Training Set (2010-2020)
| Strategy | Sharpe Ratio | Final Value | Max Drawdown |
|----------|-------------|-------------|--------------|
| HODL | 1.70 | $5.52B | -93.07% |
| DCA | 1.78 | $2.31B | -93.07% |
| Quant | 1.98 | $2.76B | - |

### Test Set (2023-2024)
| Strategy | Sharpe Ratio | Return | Final Value |
|----------|-------------|--------|-------------|
| HODL | 2.03 | +273% | $48,457 |
| DCA | 3.04 | +141% | $31,328 |
| Quant | 1.08 | +45% | $18,883 |

**Key Finding**: The quantitative strategy shows significant overfitting, with test set performance far below the training set.

### 📊 Quantitative Strategy Trading Statistics

#### Training Set (2010-2020)
- **Total Trades**: 2,783
- **Trade Frequency**: 76.73% (Avg every 1.3 days)
- **Annual Trades**: 280.3
- **Avg Trade Size**: 14.4% position change
- **Total Turnover**: 40,417.9%

#### Test Set (2023-2024)
- **Total Trades**: 296
- **Trade Frequency**: 80.87% (Avg every 1.2 days)
- **Annual Trades**: 295.4
- **Avg Trade Size**: 16.4% position change
- **Total Turnover**: 4,883.3%

**Strategy Characteristics**: High-frequency trading strategy where transaction costs significantly impact returns.

## 🗂️ Project Structure

```
bitcoin-investment-strategies/
├── src/                             # Source code folder
│   ├── strategies/                  # Strategy implementations
│   │   ├── __init__.py             # Strategy module init
│   │   ├── hodl.py                 # HODL strategy implementation
│   │   ├── dca.py                  # DCA strategy implementation (training)
│   │   ├── dca_test.py             # DCA test set evaluation
│   │   ├── quant_rf.py             # Quant strategy implementation (training)
│   │   └── quant_test.py           # Quant test set evaluation
│   ├── __init__.py                 # Package init
│   ├── config.py                   # Global config (paths, params, constants)
│   ├── metrics.py                  # Unified performance metrics calculation
│   ├── utils.py                    # Common utility functions
│   └── main.py                     # Main program (train + test + cross-validation)
├── data/                            # Data folder
│   ├── bitcoin_train_2010_2020 copy.csv  # Training set data
│   ├── bitcoin_test_2023_2024 copy.csv   # Test set data
│   └── bitcoin_valid_2021_2022 copy.csv  # Validation set data
├── docs/                            # Documentation folder
│   ├── README.md                    # Project documentation
│   ├── strategy_comparison_report.md  # Complete comparison report
│   ├── quantitative_strategy_summary.md # Quant strategy technical details
│   └── dca_and_hodl_assumptions.md    # Strategy assumptions based on code
├── run.py                           # Project entry point
├── .gitignore                       # Git ignore config
└── README.md                        # Root directory explanation
```

### Code Architecture Design

**Modular Design**:
- `config.py`: Single source of configuration, containing all paths, parameters, and pre-trained weights.
- `metrics.py`: Unified financial metric calculation (Sharpe, Sortino, Max Drawdown, etc.).
- `utils.py`: Common functions for data loading, formatting, display, etc.
- `strategies/`: Strategy implementation module, all strategies use unified configuration and metrics.

**Professional Structure**:
- `src/` folder: Follows Python best practices, centrally managing all source code.
- `run.py`: Clear project entry point, automatically configuring Python path.
- Package Initialization: Uses `__init__.py` to establish correct package structure.

## 🚀 Quick Start

### Requirements
- Python 3.8+
- pandas, numpy, scikit-learn

### Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### Run Strategies

**Run Full Analysis (Recommended)**
```bash
python run.py
```

**Output Content**:
1. Training Set Evaluation (2010-2020) - Development phase performance of three strategies.
2. Test Set Evaluation (2023-2024) - Validation of strategy generalization ability.
3. Cross-Validation Analysis - Detecting overfitting, comparing training vs test performance.
4. Final Recommendation - Strategy recommendation based on full analysis.

**Or Run Main Program Directly (Equivalent)**
```bash
python -m src.main
```

**Or Run Each Strategy Module Separately:**

```bash
# HODL Strategy (Training)
python -m src.strategies.hodl

# DCA Strategy (Training)
python -m src.strategies.dca

# DCA Strategy (Test)
python -m src.strategies.dca_test

# Quant Strategy (Training)
python -m src.strategies.quant_rf

# Quant Strategy (Test)
python -m src.strategies.quant_test
```

## 📝 Documentation Guide

All documentation is located in the `docs/` folder:

### 1. [strategy_comparison_report.md](docs/strategy_comparison_report.md)
- Detailed specific performance comparison between training and test sets
- In-depth analysis of overfitting issues
- Investment advice and risk warnings

### 2. [quantitative_strategy_summary.md](docs/quantitative_strategy_summary.md)
- Detailed explanation of 10 technical factors
- Random walk optimization algorithm principle
- Economic meaning interpretation of weights
- Strategy limitations and improvement directions

### 3. [dca_and_hodl_assumptions.md](docs/dca_and_hodl_assumptions.md)
- Core assumptions extracted based on code implementation
- Strategy parameter settings and economic logic
- Common assumptions and difference comparison

## 🔬 Quantitative Strategy Technical Details

### Ten Factors
1. **Momentum**: MOM_20, MOM_60
2. **MA Spread**: MA_50_SPREAD, MA_90_SPREAD
3. **Volatility**: VOL_20, ATR_PCT_14, VOL_RATIO_20
4. **Price Position**: PRICE_POS_60, CLOSE_POS
5. **Market Cycle**: POST_HALVING

### Optimization Method
- Random Walk Search (300 iterations, conservative optimization)
- Objective Function: Maximize Sharpe Ratio - L2 Regularization Penalty (0.012)
- Rolling Z-score Standardization (90-day window)
- Sigmoid Position Mapping (Supports 0-100% continuous position)
- Transaction Cost: 15 bps (0.15%)

### Final Weights
```python
[0.304, -1.871, 0.386, 0.368, -0.352, 1.320, 0.412, 2.561, 0.612, -0.793]
```

### Factor Weight Distribution

**Top Three Important Factors** (by absolute value):
1. **PRICE_POS_60 (60-day Price Position)** - Weight: +2.561 (28.5%)
   - Bull Signal: The closer the price is to the 60-day high, the more inclined to full position.
   
2. **MOM_60 (60-day Momentum)** - Weight: -1.871 (20.8%)
   - Bear Signal: Reverse momentum strategy vs chasing highs.
   
3. **ATR_PCT_14 (14-day ATR)** - Weight: +1.320 (14.7%)
   - Bull Signal: Increase position when volatility rises.

**Weight Distribution Statistics**:
- Bull Factors (Positive Weight): 7
- Bear Factors (Negative Weight): 3
- Strategy Bias: Overall long configuration

**Factor Weight Ratios**:
| Factor | Weight | Abs Value Share | Direction |
|--------|--------|-----------------|-----------|
| PRICE_POS_60 | +2.561 | 28.5% | Bull |
| MOM_60 | -1.871 | 20.8% | Bear |
| ATR_PCT_14 | +1.320 | 14.7% | Bull |
| RSI_14 | -0.793 | 8.8% | Bear |
| CLOSE_POS | +0.612 | 6.8% | Bull |
| VOL_RATIO_20 | +0.412 | 4.6% | Bull |
| MA_50_SPREAD | +0.386 | 4.3% | Bull |
| MA_90_SPREAD | +0.368 | 4.1% | Bull |
| VOL_20 | -0.352 | 3.9% | Bear |
| MOM_20 | +0.304 | 3.4% | Bull |

**Weight Interpretation**:
- **Price Position Dominance**: PRICE_POS_60 has the highest share, indicating the strategy mainly relies on price position within historical range.
- **Reverse Momentum**: MOM_60 has negative weight, strategy tends to reduce position when momentum is too strong.
- **Volatility Sensitivity**: ATR and VOL factors have significant weights, strategy is sensitive to market volatility.

### Trading Characteristics
- **Strategy Type**: High Frequency Trading (Daily rebalancing)
- **Training Set Avg Trades**: 280/year
- **Test Set Avg Trades**: 295/year
- **Avg Position Adjustment**: 14-16%
- **Turnover Rate**: Extremely high (Training set 40,417%, Test set 4,883%)

## ⚠️ Risk Warning

1. **Overfitting Risk**: Quant strategy Sharpe dropped from 1.98 to 1.08 in test set, significant performance decline.
2. **High Frequency Costs**: 280+ trades/year, transaction costs have huge impact on returns.
3. **Historical Data Limitations**: Past performance does not represent future returns.
4. **Market Risk**: Extreme events may cause strategy failure.
5. **Technical Risk**: Blockchain technology and regulatory environment changes.

## 📊 Core Metrics

- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Maximum loss from peak
- **Risk-Free Rate**: 3% (Annualized)
- **Capital Cost**: 7% (HODL strategy)

## 💡 Key Conclusion

> Simple DCA strategy significantly outperforms overly optimized quantitative strategies in long-term performance and robustness.

In Bitcoin investment:
- ✅ **DCA Strategy Optimal**: Robust performance in both training and test sets, highest risk-adjusted return.
- ✅ **HODL Strategy for Large Capital**: High absolute return, but must withstand higher volatility.
- ⚠️ **Quant Strategy Caution**: High frequency costs (280+ trades/year), significant overfitting risk.
- 📊 **Transaction Costs Critical**: Quant strategy's 40,000%+ turnover makes transaction costs a profit killer.

## 📚 References

- Bitcoin Historical Price Data (2010-2024)
- Modern Portfolio Theory
- Technical Analysis Indicators
- Random Walk Optimization

## 🤝 Contribution

Issues and Pull Requests are welcome to improve strategy implementation.

## 📄 License

MIT License

---

**Disclaimer**: This project is for academic research and educational purposes only and does not constitute investment advice. Investment involves risks, please be cautious.
