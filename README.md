# Bitcoin Investment Strategies: HODL vs DCA vs Quantitative

A comparative analysis project implementing and comparing three classic investment strategies for Bitcoin from 2010 to 2024.

## 🚀 Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run full analysis
python run.py
```

## 📊 Three Strategies

1. **HODL** (Buy and Hold) - Invest $13,000 at once, hold for long term.
2. **DCA** (Dollar-Cost Averaging) - Invest $1,000 monthly for 13 months (Total $13,000).
3. **Quantitative** - Quantitative trading system based on 10 technical factors.

## 📈 Core Results

### Test Set Performance (2023-2024)
| Strategy | Sharpe Ratio | Final Value | Performance |
|----------|-------------|-------------|-------------|
| **DCA** | **3.04** | $31,328 | ✅ Best |
| **HODL** | **2.03** | $48,457 | ✅ Highest Return |
| Quant | 1.08 | $18,883 | ⚠️ Overfitting |

### Key Findings
- ✅ **DCA Strategy**: Robust and reliable, excellent risk-adjusted returns.
- ✅ **HODL Strategy**: Highest absolute return, strong Sharpe ratio.
- ⚠️ **Quantitative Strategy**: Training set Sharpe 1.98 → Test set 1.08, significant overfitting.
- 📊 **High Frequency Costs**: Quantitative strategy averages 280+ trades/year, 40,000%+ turnover rate.

## 📊 Visualizations

The project includes various professional visualizations:

### Dynamic Display
- 📈 **Portfolio Growth Animation** (GIF)
  - Training set (2010-2020) and Test set (2023-2024)
  - Step-by-step display of value changes for three strategies
  - Loop playback for intuitive comparison

### Static Charts
- 📊 **Portfolio Value Curve** - Comparison of Training and Test sets
- 📍 **Position Change Time Series** - Visualization of dynamic rebalancing in quantitative strategy
- 📈 **Cumulative Trade Count** - Showing trade frequency and turnover rate
- 🥧 **Factor Weight Distribution** - Importance of 10 technical factors

All visualization files are located in the `visualization/` folder.

## 🗂️ Project Structure

```
bitcoin-investment-strategies/
├── src/                    # Source code
│   ├── strategies/        # Strategy implementations
│   ├── config.py         # Global configuration
│   ├── metrics.py        # Performance metrics
│   ├── utils.py          # Utility functions
│   └── main.py           # Main program
├── data/                  # Data files
├── docs/                  # Detailed documentation
├── visualization/         # Visualization charts and scripts
│   ├── *.png             # Static charts
│   ├── *.gif             # Dynamic GIFs
│   └── plot_*.py         # Plotting scripts
└── run.py                # Entry point
```

## 📚 Detailed Documentation

Please refer to [`docs/README.md`](docs/README.md) for full technical documentation and analysis reports.

Includes:
- Detailed strategy implementation explanation
- Complete performance comparison analysis
- Quantitative strategy technical details
- Code architecture design document

## 🛠️ Tech Stack

- **Python 3.13**
- **pandas** - Data processing
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **imageio** - GIF animation generation

## 📄 License

MIT License

---

**Author**: lucky11chances
**GitHub**: [bitcoin-investment-strategies](https://github.com/lucky11chances/bitcoin-investment-strategies)
