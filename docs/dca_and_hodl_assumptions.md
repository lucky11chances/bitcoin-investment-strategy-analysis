# HODL and DCA Strategy Code Implementation Assumptions Summary

## Strategy Parameter Settings

### HODL Strategy (hodl.py)
```python
INITIAL_CAPITAL = 12_500.0        # One-time investment of $12,500
RISK_FREE_ANNUAL = 0.03           # 3% Annualized Risk-Free Rate (for Sharpe/Sortino)
CAPITAL_COST_ANNUAL = 0.07        # 7% Annualized Capital Cost (Opportunity Cost)
DAYS_PER_YEAR = 365
```

### DCA Strategy (dca.py)
```python
MONTHLY_CONTRIBUTION = 1_000.0    # Monthly fixed investment of $1,000
TARGET_TOTAL = 12_500.0           # Total investment target $12,500 (13 months)
RISK_FREE_ANNUAL = 0.03           # 3% Annualized Risk-Free Rate
DAYS_PER_YEAR = 365
```

---

## I. HODL Strategy Core Assumptions (Based on Code)

### Assumption 1: Timing is invalid, one-time full position buying is optimal
**Code Reflection**:
```python
buy_price = close.iloc[0]        # Buy on the first day of the dataset
btc_held = INITIAL_CAPITAL / buy_price  # Convert all funds to BTC
```
- No batch buying, no waiting for timing.
- Assumes buying on the first day is no different from buying at any other time (in the long run).

### Assumption 2: Zero operation during holding period
**Code Reflection**:
```python
portfolio = close * btc_held     # Only calculate market value daily
daily_ret = portfolio.pct_change().dropna()  # No active trading
```
- BTC quantity remains constant (`btc_held` fixed).
- No stop-loss, no take-profit, no rebalancing.

### Assumption 3: Importance of risk-adjusted returns
**Code Reflection**:
```python
rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / DAYS_PER_YEAR) - 1.0
excess_ret = daily_ret - rf_daily
sharpe = (excess_ret.mean() / vol) * math.sqrt(DAYS_PER_YEAR)
```
- Uses 3% risk-free rate to calculate excess returns.
- Assumes investors care about "excess return per unit of risk", not just absolute return.

### Assumption 4: Asymmetry between downside risk and upside volatility
**Code Reflection**:
```python
sortino = np.nan
downside = excess_ret[excess_ret < 0]  # Only calculate negative returns
down_vol = downside.std(ddof=1)
sortino = (excess_ret.mean() / down_vol) * math.sqrt(DAYS_PER_YEAR)
```
- Uses Sortino instead of just Sharpe, indicating more focus on downside risk.
- Assumes investors' aversion to loss is higher than their preference for profit.

### Assumption 5: Opportunity cost needs quantification
**Code Reflection**:
```python
capital_cost = INITIAL_CAPITAL * (1.0 + CAPITAL_COST_ANNUAL) ** duration_years
net_profit_after_cost = final_value - capital_cost
```
- Assumes funds have a 7% annualized opportunity cost (could be loan interest or other investment returns).
- Final profit needs to deduct this implicit cost.

### Assumption 6: Extreme drawdown is tolerable
**Code Reflection**:
```python
running_max = portfolio.cummax()
drawdown = portfolio / running_max - 1.0
max_drawdown = float(drawdown.min())  # Record max drawdown but trigger no action
```
- Calculating max drawdown is for statistical purposes only, no stop-loss line set.
- Assumes holding can persist even with 90%+ loss.

---

## II. DCA Strategy Core Assumptions (Based on Code)

### Assumption 1: Regular fixed amount is better than one-time investment
**Code Reflection**:
```python
monthly_first = df.resample("MS", on="Date").first().reset_index()
monthly_first["btc_bought"] = MONTHLY_CONTRIBUTION / monthly_first["Close"]
```
- Buy fixed $1,000 on the first trading day of each month.
- Assumes spreading buy times can reduce timing risk.

### Assumption 2: Price volatility automatically achieves buy low sell high
**Code Reflection**:
```python
btc_bought = MONTHLY_CONTRIBUTION / monthly_first["Close"]
```
- Fixed Dollar Amount ÷ Current Price = BTC Quantity Bought
- **Low Price**: Small denominator → Buy more BTC (Auto bottom fishing)
- **High Price**: Large denominator → Buy less BTC (Auto reduce position)
- No subjective judgment needed, price automatically adjusts position.

### Assumption 3: Investment cycle can be preset
**Code Reflection**:
```python
max_months = int(np.ceil(TARGET_TOTAL / MONTHLY_CONTRIBUTION))  # 13 months
monthly_first = monthly_first.head(max_months)  # Execute exactly 13 times
```
- Stop after investing $12,500, same total input as HODL.
- Assumes investors have clear capital planning and discipline.

### Assumption 4: Initial zero return period needs exclusion
**Code Reflection**:
```python
invested = merged[merged["cum_btc"] > 0].copy()  # Only calculate after holding BTC
daily_ret = invested["portfolio"].pct_change().dropna()
```
- Return is 0 before first buy, pct_change would generate inf/nan.
- Assumes return calculation should start from having a position.

### Assumption 5: Cumulative position increases gradually
**Code Reflection**:
```python
monthly_first["cum_btc"] = monthly_first["btc_bought"].cumsum()
merged["cum_btc"] = merged["cum_btc"].ffill().fillna(0.0)  # Forward fill
```
- BTC bought each time is held permanently, never sold.
- Holdings increase monotonically over time.

### Assumption 6: Risk assessment consistent with HODL
**Code Reflection**:
```python
rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / DAYS_PER_YEAR) - 1.0
sharpe = (excess_ret.mean() / vol) * math.sqrt(DAYS_PER_YEAR)
sortino = (excess_ret.mean() / down_vol) * math.sqrt(DAYS_PER_YEAR)
```
- Uses same 3% risk-free rate.
- Same Sharpe/Sortino calculation logic.
- Assumes both strategies are compared under the same risk framework.

---

## III. Common Assumptions of Both Strategies

### 1. Data Integrity Assumption
```python
# Both assume CSV file exists and format is correct
df = pd.read_csv(path)
df[date_col] = pd.to_datetime(df[date_col])
```
- Date column automatically identified (Date or Start).
- Close price continuous without missing values.
- Data sorted ascending by time.

### 2. Compound Interest Calculation Assumption
```python
# Both convert annualized rate to daily frequency
rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / 365) - 1.0
```
- Assumes risk-free rate compounds, not simple interest.
- One year fixed at 365 days (leap years ignored).

### 3. Return Standardization Assumption
```python
# Both annualize Sharpe/Sortino
sharpe = (excess_ret.mean() / vol) * math.sqrt(365)
```
- Assumes daily returns can be annualized by multiplying √365.
- Requires assumption that returns are IID (Independent and Identically Distributed).

### 4. Zero Transaction Cost Assumption
```python
# Neither strategy deducts transaction fees
btc_held = INITIAL_CAPITAL / buy_price  # HODL
btc_bought = MONTHLY_CONTRIBUTION / Close  # DCA
```
- No fees, slippage, taxes.
- Assumes 100% of funds convert to BTC.

### 5. Infinite Liquidity Assumption
```python
# Both assume buying at any price is possible
portfolio = close * btc_held
```
- Market depth sufficient, buy orders won't push up price.
- Executable at close price at any time point.

---

## IV. Comparison of Strategy Difference Assumptions

| Dimension | HODL Assumption | DCA Assumption |
|-----------|-----------------|----------------|
| **Capital Allocation** | Assumes large lump-sum capital available | Assumes stable monthly cash flow available |
| **Timing Ability** | Assumes buying on day 1 equals any time point | Assumes timing impossible, spread risk via time |
| **Risk Tolerance** | Assumes -93% max drawdown tolerable | Assumes batch buying reduces psychological pressure |
| **Operational Discipline** | Assumes zero operation after buy | Assumes strict execution of 13 fixed investments |
| **Cost Consideration** | Considers 7% capital cost (opportunity cost) | Ignores capital cost, calculates return only |

---

## V. Implicit Market Views in Code

### HODL Implicit View
1. **Long-term holding beats frequent trading** (Otherwise should set stop-loss/take-profit)
2. **Volatility is noise, trend is signal** (Otherwise should avoid high volatility periods)
3. **Time is the best friend** (Longer holding time is better)

### DCA Implicit View
1. **Short-term price unpredictable** (Otherwise should concentrate buy at lows)
2. **Dollar Cost Averaging works** (Otherwise should invest lump-sum)
3. **Disciplined execution beats subjective judgment** (Mechanized monthly execution)

---

## Summary

The code implementations of these two strategies reveal their core assumptions:

**HODL = "I believe the long-term trend is up, short-term volatility doesn't matter, and I have enough capital and patience"**

**DCA = "I cannot predict price, but I have stable cash flow, spreading risk over time is safest"**

Both assume:
- Bitcoin appreciates long-term
- Active timing is invalid
- Simple execution beats complex strategies
- Transaction costs negligible
- Market liquidity sufficient

Differences lie in:
- HODL assumes investor has lump-sum capital and extreme risk tolerance.
- DCA assumes investor cares more about psychological comfort and risk smoothing.

---

**Data Source**: hodl.py and dca.py code implementation
**Document Date**: December 12, 2025
