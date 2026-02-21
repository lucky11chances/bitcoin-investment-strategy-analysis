import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="BTC Strategy Analysis Report",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Sidebar: Experimental Setup & Authors
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Experimental Setup")
    st.info("Parameters are fixed to match Section 1.2 of the report.")
    
    st.markdown("### 🔹 HODL Strategy")
    st.markdown("""
    * **Principal:** $13,000 (Lump Sum)
    * **Action:** One-time injection
    * **Timing:** Start of Test Period
    """)
    
    st.divider()
    
    st.markdown("### 🔹 DCA Strategy")
    st.markdown("""
    * **Total Principal:** $13,000
    * **Action:** $1,000 / month
    * **Duration:** 13 Months
    """)

    st.divider()
    
    st.markdown("### 🔹 Quant Strategy")
    st.markdown("""
    * **Principal:** $13,000 (Base)
    * **Optimization:** Sharpe Ratio
    * **Cost:** 0.15% per trade
    """)

    st.divider()
    
    # --- 👥 Authors Section in Sidebar ---
    st.markdown("### 👨‍🎓 Project Team")
    st.markdown("""
    **Haoyu Xie** **Xiangyu Yue** **Linxiao Chen** **Ruoxuan Huang**
    """)
    
    st.caption("Data Source: Comparative Analysis of Bitcoin Investment Strategies (2010–2024)")

# ---------------------------------------------------------
# 3. Main Header & Authors (Top of Page)
# ---------------------------------------------------------
st.title("📊 Comparative Analysis of Bitcoin Investment Strategies (2010–2024)")

# --- ✍️ Authors Display ---
st.markdown("""
<style>
.author-text {
    font-size: 18px;
    font-weight: 500;
    color: #4a4a4a;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    **👨‍🎓 Project Team:** Haoyu Xie, Xiangyu Yue, Linxiao Chen, Ruoxuan Huang  
    *A Data Science Project evaluating HODL, DCA, and Quantitative Trading Strategies.*
    """
)
st.markdown("---")

# ---------------------------------------------------------
# 4. Dashboard Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 Executive Overview", 
    "📈 Performance Analysis", 
    "🧠 Quant Deep Dive", 
    "🏁 Conclusion"
])

# --- Tab 1: Executive Overview ---
with tab1:
    st.header("1. Executive Overview")
    st.markdown("""
    **Objective:** This report evaluates the historical performance and risk profiles of three distinct Bitcoin investment strategies. 
    The study utilizes data spanning 2010–2024, with a specific focus on the **Testing Set (2023–2024)**.
    """)
    
    # KPIs - STRICTLY FROM SECTION 1.3 Key Findings Table
    st.subheader("🏆 Key Findings: Testing Phase (2023-2024)")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown("### 🏦 HODL")
        st.metric(
            label="Final Value ($13,000 Start)",
            value="$48,457",
            delta="Highest Return"
        )
        st.caption("**Sharpe Ratio: 2.03** | Max Drawdown: -93.07%")

    with kpi2:
        st.markdown("### 📅 DCA")
        st.metric(
            label="Final Value ($13,000 Total)",
            value="$31,328",
            delta="Most Robust"
        )
        st.caption("**Sharpe Ratio: 3.04** (Best Risk-Adjusted)")

    with kpi3:
        st.markdown("### 🤖 Quant")
        st.metric(
            label="Final Value ($13,000 Start)",
            value="$18,883",
            delta="Underperformed",
            delta_color="inverse"
        )
        st.caption("**Sharpe Ratio: 1.08** (Severe Overfitting)")

    st.markdown("### 📉 Strategy Comparison Matrix (Section 1.3)")
    # Data strictly from Section 1.3 Table
    comparison_data = {
        "Metric": ["Training Sharpe (2010–2020)", "Testing Sharpe (2023–2024)", "Testing ROI", "Final Value (Testing)"],
        "HODL ($13k)": ["1.70", "2.03", "268%", "$48,457"],
        "DCA ($13k)": ["1.78", "3.04", "141%", "$31,328"],
        "Quant ($13k)": ["1.98", "1.08", "45%", "$18,883"]
    }
    st.table(pd.DataFrame(comparison_data).set_index("Metric"))

# --- Tab 2: Performance Analysis ---
with tab2:
    st.header("3. Performance & Equity Curves")
    
    # Image Base URL
    BASE_URL = "https://raw.githubusercontent.com/lucky11chances/bitcoin-investment-strategies-draft/main/visualization"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Period (2010-2020)")
        st.image(f"{BASE_URL}/portfolio_value_training.png", caption="Fig 1. In-Sample Performance")
        st.info("""
        **Training Phase:** * The Quant strategy (Orange) achieved the highest Sharpe Ratio (1.98).
        * HODL reached the highest absolute value but with extreme volatility.
        """)
        
    with col2:
        st.subheader("Testing Period (2023-2024)")
        st.image(f"{BASE_URL}/portfolio_value_test.png", caption="Fig 2. Out-of-Sample Performance")
        st.error("""
        **Testing Phase (Overfitting):** * **Quant (Orange)** flatlined at $18,883.
        * **HODL (Blue)** rallied to $48,457.
        * **DCA (Purple)** provided steady growth to $31,328.
        """)

# --- Tab 3: Quant Deep Dive ---
with tab3:
    st.header("2. Technical Deep Dive: Why Quant Failed")
    
    st.markdown("""
    The Quant strategy utilized a **Random Walk Optimization** on 10 technical factors (Momentum, Trend, Volatility, etc.).
    Despite a strong training score, it failed to generalize.
    """)
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("### 3.1 Factor Importance")
        st.image(f"{BASE_URL}/factor_weights_en.png", use_container_width=True)
        st.caption("Fig 3. Weight Distribution")
        st.markdown("""
        **Key Findings:**
        * **PRICE_POS_60 (28.5%)**: The primary driver.
        * **MOM_60**: Strong negative weight (Counter-trend bias).
        """)

    with c2:
        st.markdown("### 3.3 Trading Frequency & Cost")
        st.image(f"{BASE_URL}/cumulative_trades.png", use_container_width=True)
        st.caption("Fig 4. Cumulative Trades")
        st.warning("""
        **The Cost of Churn:**
        * **Turnover:** 4,883% annual turnover.
        * **Trades:** 296 trades in the test set alone.
        * **Impact:** 0.15% fee per trade eroded the 45% gross return.
        """)

    st.divider()
    st.markdown("### 3.2 Market Timing (Position Changes)")
    st.image(f"{BASE_URL}/position_changes.png", use_container_width=True)
    st.caption("Fig 5. Binary Position Switching (0% vs 100%)")
    st.markdown("The model exhibited 'nervous' behavior, frequently switching between full BTC position and Cash, indicating a lack of consistent trend identification in the test set.")

# --- Tab 4: Conclusion ---
with tab4:
    st.header("4. Results and Conclusion")
    
    st.success("""
    ### ✅ Final Verdict: DCA is the Superior Strategy
    
    > "Adopting a Dollar-Cost Averaging (DCA) approach is recommended."
    
    Based on the Comparative Analysis (2010–2024):
    
    1.  **Robustness (Winner: DCA):** DCA achieved a **Sharpe Ratio of 3.04** in the testing phase. It mitigates timing risk and psychological stress.
    2.  **Overfitting (Loser: Quant):** The Quant model's Sharpe Ratio collapsed from **1.98 (Train)** to **1.08 (Test)**. It memorized historical noise but failed in the new market regime.
    3.  **Risk (HODL):** While HODL had the highest final value (**$48,457**), it carries a historical max drawdown of **-93.07%**, making it unsuitable for risk-averse investors.
    """)
    
    st.markdown("---")
    st.caption("References: lucky11chances/bitcoin-investment-strategies-draft (GitHub) | Data: Bitcoin Historical Price Data (2010–2024)")
