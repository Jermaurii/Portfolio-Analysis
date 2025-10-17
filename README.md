# 📈 Real-Life Portfolio Factor Analysis

A comprehensive portfolio analysis tool that uses real stock data and the Fama-French five-factor model (plus momentum) to provide detailed insights into portfolio performance and risk characteristics.

## ✨ Features

### 🏆 New: Real-Life Stock Analysis
- **Real Stock Data**: Analyze actual stocks using live market data from Yahoo Finance
- **Interactive Stock Selection**: Choose from popular stock sectors or enter custom tickers
- **Accurate Factor Models**: Uses Fama-French five-factor + momentum model with real factor data
- **Detailed Explanations**: Get comprehensive interpretations of every metric and number

### 📊 Comprehensive Analysis
- **Factor Exposure Analysis**: Understand exposure to market, size, value, momentum, profitability, and investment factors
- **Alpha Generation**: Measure portfolio skill beyond factor exposures
- **Attribution Analysis**: See which factors drive portfolio returns
- **Rolling Analysis**: Track how exposures change over time
- **Risk Assessment**: Evaluate diversification quality and risk characteristics

### 🎯 Multiple Data Sources
1. **Real Stock Portfolio**: Analyze your actual stock holdings
2. **Upload CSVs**: Use your own historical data
3. **Generate Synthetic**: Create simulated portfolios for testing
4. **Live Artificial**: Real-time simulated portfolio updates

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application (public, no keys)
```bash
python run.py real --tickers AAPL,MSFT --rolling 36

# Or launch the Streamlit app
streamlit run streamlit_app.py
```

### 3. Analyze Your Portfolio
1. Use the CLI: `python run.py real --tickers AAPL,MSFT --start 2020-01-01 --end 2024-12-31`
2. Or Select "Real Stock Portfolio" mode in Streamlit
3. Choose stocks from popular sectors or enter custom tickers
4. Set your analysis date range
5. Click "Analyze Portfolio"
5. Get detailed explanations of your results!

## 📋 Stock Selection Options

### Popular Sectors
- **Technology**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX
- **Financials**: JPM, BAC, WFC, GS, MS, C, AXP, USB
- **Healthcare**: JNJ, PFE, UNH, ABBV, MRK, TMO, ABT, DHR
- **Consumer**: KO, PEP, WMT, PG, JNJ, MCD, NKE, SBUX
- **Energy**: XOM, CVX, COP, EOG, SLB, KMI, PSX, MPC
- **Industrial**: BA, CAT, GE, UTX, HON, MMM, FDX, UPS
- **Materials**: LIN, APD, SHW, ECL, DD, PPG, NEM, FCX
- **Utilities**: NEE, SO, DUK, AEP, EXC, XEL, ES, EIX
- **REIT**: AMT, PLD, CCI, EQIX, WELL, PSA, O, SPG
- **ETFs**: SPY, QQQ, VTI, EFA, EEM, VWO, TLT, GLD

### Custom Portfolios
Enter any stock tickers separated by commas: `AAPL, GOOGL, MSFT, TSLA`

## 📈 Understanding Factor Analysis Results

### Key Metrics Explained

#### **Alpha**
- **What it means**: Excess returns beyond what factors can explain
- **Interpretation**: 
  - Positive: Portfolio shows skill/stock selection ability
  - Negative: Underperforming relative to risk taken
  - Example: α = 0.002 = 0.2% monthly excess return

#### **R-squared**
- **What it means**: How much of portfolio variance is explained by factors
- **Interpretation**:
  - >80%: Well-diversified, predictable performance
  - 60-80%: Good diversification with some unexplained variance
  - <60%: Opportunity for better diversification

#### **Factor Betas (β)**
Each factor represents exposure to different risk/return drivers:

- **MKT_RF (Market)**: Moves with overall market
  - β = 1.0: Moves exactly with market
  - β > 1.0: More volatile than market
  - β < 1.0: Less volatile than market

- **SMB (Size)**: Small-cap vs large-cap bias
  - Positive: Favors small companies
  - Negative: Favors large companies

- **HML (Value)**: Value vs growth preference
  - Positive: Value stock bias
  - Negative: Growth stock bias

- **MOM (Momentum)**: Trend-following behavior
  - Positive: Benefits from momentum
  - Negative: Contrarian/mean-reverting

- **RMW (Profitability)**: Quality company exposure
  - Positive: Targets profitable firms
  - Negative: Accepts lower profitability

- **CMA (Investment)**: Conservative vs aggressive growth
  - Positive: Conservative investment style
  - Negative: Aggressive growth approach

## 🔍 Sample Analysis Workflow

### Example: Technology Portfolio Analysis
1. **Select**: Technology sector stocks (AAPL, MSFT, GOOGL, NVDA)
2. **Analyze**: 5-year historical period
3. **Results**: 
   - Market Beta: 1.15 (more volatile than market)
   - Minimal size exposure: -0.05 (large-cap preference)
   - Positive momentum: 0.12 (momentum-driven)
   - Alpha: 0.0015 (small positive alpha)

4. **Interpretation**: 
   - Tech portfolio is aggressive (high market beta)
   - Benefits from momentum trends
   - Shows some stock-picking skill (positive alpha)
   - Well-diversified across tech stocks

## 📊 Output Files

The application generates comprehensive reports including:
- **Interactive Charts**: Betas, attribution, cumulative returns, rolling analysis
- **Data Tables**: CSV exports of all metrics and rolling betas
- **HTML Report**: Standalone portfolio analysis report
- **Explanation Report**: Detailed interpretation text file

## 🛠 Technical Details

### Factor Model
Uses the Fama-French five-factor model plus momentum:
```
Portfolio Return = α + β₁(MKT_RF) + β₂(SMB) + β₃(HML) + β₄(MOM) + β₅(RMW) + β₆(CMA) + ε
```

### Data Sources
- **Stock Data**: Stooq (via pandas-datareader)
- **Factor Data**: Kenneth French Data Library
- **Risk-Free Rate**: RF from the Kenneth French Research Factors dataset
- **Real-Time**: Updates automatically for current analysis

### Methodology
1. **Data Collection**: Fetch historical stock and factor data
2. **Portfolio Construction**: Equal-weight portfolio returns
3. **Factor Regression**: OLS regression to estimate factor exposures
4. **Analysis**: Calculate alpha, R-squared, and attribution
5. **Rolling Analysis**: Track exposures over time with moving windows; includes rolling R²
6. **Interpretation**: Generate detailed explanations and recommendations

## 🔧 Installation Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
streamlit>=1.32
scikit-learn>=1.3
yfinance>=0.2.33
requests>=2.31.0
fredapi>=0.5.1
```

## 📝 Usage Notes

- **Data Requirements**: Minimum 24 months of data recommended for reliable results
- **Stock Tickers**: Use standard Yahoo Finance symbols
- **Date Ranges**: Analysis works best with 2+ years of data
- **Performance**: Real data fetching may take 10-30 seconds depending on date range

## 🤝 Contributing

Feel free to contribute improvements:
- Additional factor models
- More data sources
- Enhanced explanations
- Performance optimization
- UI/UX enhancements

## 📄 License

This project is open source and available under the MIT License.

---

**Ready to analyze your portfolio? Start the app and dive into real-world factor analysis! 🚀**
