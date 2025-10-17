from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import time

# Make local src importable
import sys
APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

from src.config import Paths
import src.data as D
import src.model as M
import src.viz as V
import src.report as R
from src.real_data import RealDataFetcher
from src.explanations import ExplanationsGenerator


st.set_page_config(page_title="Real-Life Portfolio Factor Analysis", layout="wide")
st.title("📈 Real-Life Portfolio Factor Analysis")
st.caption("Analyze your portfolio using real stock data and accurate factor models. Choose from real stocks or upload your own data.")

# Subtle style tweaks
st.markdown(
    """
    <style>
      .stMetric { background: #fafafa; padding: 12px; border-radius: 12px; border: 1px solid #eee; }
      .chip { display:inline-block; padding:4px 10px; margin:2px; background:#eef2ff; border-radius:999px; font-size:0.85em; }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.checkbox("ℹ️ Learn About Factor Analysis", key="info"):
    st.markdown("""
    ### What is Portfolio Factor Analysis?
    
    Portfolio factor analysis uses the **Fama-French five-factor model** (plus momentum) to understand what drives your portfolio's returns:
    
    - **📊 MKT_RF (Market Risk Premium)**: How much your portfolio moves with the overall market
    - **🏢 SMB (Size Factor)**: Small-cap vs large-cap stock exposure  
    - **💰 HML (Value Factor)**: Value vs growth stock preference
    - **🚀 MOM (Momentum Factor)**: Tendency to follow recent price trends
    - **💼 RMW (Profitability Factor)**: Exposure to highly profitable companies
    - **⚖️ CMA (Investment Factor)**: Conservative vs aggressive growth companies
    
    **Alpha** represents your portfolio's excess returns beyond what factors can explain - true skill!
    
    **R-squared** shows how well factors explain your portfolio's movements (closer to 100% = more predictable).
    """)


@st.cache_data(show_spinner=False)
def _generate_synthetic(months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = D.generate_synthetic_factors(months)
    portfolio = D.generate_synthetic_portfolio(factors)
    return factors, portfolio


def _read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, parse_dates=["date"], index_col="date")


with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose input method", ["Real Stock Portfolio", "Upload CSVs", "Generate synthetic", "Live artificial portfolio"], index=0)
    rolling_window = st.number_input("Rolling window (months, 0 = off)", min_value=0, max_value=240, value=36)

    real_opts = None
    if mode == "Real Stock Portfolio":
        st.subheader("Stock Selection")
        
        # Get data fetcher instance
        fetcher = RealDataFetcher()
        popular_stocks = fetcher.get_popular_stocks()
        
        # Sector selection
        selected_sector = st.selectbox("Choose sector:", ["Custom Portfolio"] + list(popular_stocks.keys()))
        
        # Stock selection
        portfolio_stocks = []
        
        if selected_sector != "Custom Portfolio":
            portfolio_stocks = popular_stocks[selected_sector]
            st.info(f"Selected: {selected_sector} stocks")
            if st.button(f"Use {selected_sector} Portfolio"):
                portfolio_stocks = popular_stocks[selected_sector]
        else:
            # Custom stock entry
            stock_input = st.text_input(
                "Enter stock symbols (comma-separated)", 
                placeholder="AAPL, GOOGL, MSFT, TSLA",
                help="Enter stock tickers separated by commas"
            )
            
            if stock_input:
                portfolio_stocks = [s.strip().upper() for s in stock_input.split(',') if s.strip()]
        
        # Display selected portfolio
        if portfolio_stocks:
            st.write("**Selected Portfolio:**")
            st.markdown(" ".join([f"<span class='chip'>{s}</span>" for s in portfolio_stocks]), unsafe_allow_html=True)
        
        # Portfolio options
        st.subheader("Analysis Options")
        start_date = st.date_input(
            "Start Date", 
            value=pd.Timestamp.now() - pd.Timedelta(days=1825),  # ~5 years
            min_value=pd.Timestamp('2010-01-01'),
            max_value=pd.Timestamp.now()
        )
        
        end_date = st.date_input(
            "End Date",
            value=pd.Timestamp.now(),
            min_value=pd.Timestamp('2010-01-01'),
            max_value=pd.Timestamp.now()
        )
        
        if portfolio_stocks and len(portfolio_stocks) >= 1:
            real_opts = {
                'stocks': portfolio_stocks,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
        
        st.info("Tip: You can also set dates and tickers in a .env file and use the CLI.")
        run_btn = st.button("Analyze Portfolio", type="primary", disabled=not portfolio_stocks)
    else:
        run_btn = st.button("Run analysis", type="primary")

    live_opts = None
    if mode == "Live artificial portfolio":
        st.subheader("Live options")
        live_seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
        live_window_pts = st.number_input("Analysis window (points)", min_value=24, max_value=2000, value=120, step=12)
        live_sigma = st.slider("Volatility scale", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        live_opts = dict(seed=live_seed, window=live_window_pts, sigma=live_sigma)
        run_btn = True  # always compute in live mode


factors_df: pd.DataFrame | None = None
portfolio_df: pd.DataFrame | None = None

if mode == "Real Stock Portfolio":
    st.subheader("Real Stock Analysis")
    
    if real_opts is not None:
        try:
            with st.spinner("Fetching real stock data and factors..."):
                # Initialize data fetcher
                fetcher = RealDataFetcher()
                
                # Get factor data
                st.info("📊 Loading factor data...")
                factors_df = fetcher.get_complete_factor_data(real_opts['start_date'], real_opts['end_date'])
                
                # Get portfolio data
                st.info("📈 Loading stock data...")
                portfolio_df = fetcher.create_portfolio_data(
                    symbols=real_opts['stocks'],
                    start_date=real_opts['start_date'],
                    end_date=real_opts['end_date']
                )
                
                # Align data on index only (keep distinct columns)
                common_idx = factors_df.index.intersection(portfolio_df.index)
                factors_df = factors_df.loc[common_idx]
                portfolio_df = portfolio_df.loc[common_idx]
                
                st.success(f"✅ Loaded data: {factors_df.shape[0]} monthly periods")
                
        except Exception as e:
            st.error(f"Error loading real data: {e}")
            st.info("💡 Try with fewer stocks or a shorter date range")

elif mode == "Upload CSVs":
    st.subheader("Upload Files")
    f1, f2 = st.columns(2)
    with f1:
        up_factors = st.file_uploader("Factors CSV (must include columns MKT_RF, SMB, HML, MOM, RMW, CMA, RF)", type=["csv"], key="factors")
    with f2:
        up_port = st.file_uploader("Portfolio CSV (must include columns Portfolio, RF)", type=["csv"], key="portfolio")
    if up_factors is not None:
        try:
            factors_df = _read_csv(up_factors)
            st.success(f"Loaded factors: {factors_df.shape[0]} rows")
            st.dataframe(factors_df.head())
        except Exception as e:
            st.error(f"Error reading factors CSV: {e}")
    if up_port is not None:
        try:
            portfolio_df = _read_csv(up_port)
            st.success(f"Loaded portfolio: {portfolio_df.shape[0]} rows")
            st.dataframe(portfolio_df.head())
        except Exception as e:
            st.error(f"Error reading portfolio CSV: {e}")
elif mode == "Generate synthetic":
    st.subheader("Synthetic Data")
    months = st.slider("Months of synthetic data", 24, 240, 120, 12)
    if st.button("Generate synthetic data"):
        factors_df, portfolio_df = _generate_synthetic(months)
        st.session_state["factors_df"] = factors_df
        st.session_state["portfolio_df"] = portfolio_df
    # Persist for display
    factors_df = st.session_state.get("factors_df")
    portfolio_df = st.session_state.get("portfolio_df")
    if factors_df is not None and portfolio_df is not None:
        st.success(f"Generated: {factors_df.shape[0]} months")
        st.dataframe(factors_df.head())
        st.dataframe(portfolio_df.head())
elif mode == "Live artificial portfolio":
    st.subheader("Live artificial portfolio (updates every second)")

    # Initialize live state
    if "live_initialized" not in st.session_state:
        rng = np.random.default_rng(42)
        base = D.generate_synthetic_factors(60, seed=42)
        base_port = D.generate_synthetic_portfolio(base)
        st.session_state["live_factors"] = base
        st.session_state["live_portfolio"] = base_port
        st.session_state["live_initialized"] = True

    # Append a new synthetic tick
    if live_opts is not None:
        rng = np.random.default_rng(int(live_opts["seed"]))
        last_idx = st.session_state["live_factors"].index.max()
        t_next = pd.Timestamp.now()
        # Scale vol with sigma; monthly-like step but arbitrary per-second
        specs = {
            "MKT_RF": (0.005, 0.04 * live_opts["sigma"]),
            "SMB": (0.002, 0.03 * live_opts["sigma"]),
            "HML": (0.003, 0.025 * live_opts["sigma"]),
            "MOM": (0.004, 0.035 * live_opts["sigma"]),
            "RMW": (0.0025, 0.03 * live_opts["sigma"]),
            "CMA": (0.002, 0.03 * live_opts["sigma"]),
        }
        row = {k: rng.normal(mu, sigma) for k, (mu, sigma) in specs.items()}
        row["RF"] = 0.00025
        f_new = pd.DataFrame([row], index=[t_next])
        f_new.index.name = "date"
        st.session_state["live_factors"] = pd.concat([st.session_state["live_factors"], f_new])

        # Build portfolio using same generator logic
        alpha = 0.0008
        betas = {"MKT_RF": 1.05, "SMB": 0.25, "HML": -0.15, "MOM": 0.40, "RMW": 0.10, "CMA": -0.05}
        eps = rng.normal(0.0, 0.01)
        excess = alpha + sum(betas[k] * f_new.iloc[0][k] for k in betas) + eps
        total = excess + f_new.iloc[0]["RF"]
        p_new = pd.DataFrame({"Portfolio": [total], "RF": [f_new.iloc[0]["RF"]]}, index=[t_next])
        p_new.index.name = "date"
        st.session_state["live_portfolio"] = pd.concat([st.session_state["live_portfolio"], p_new])

        # Keep only window points
        win = int(live_opts["window"]) if live_opts["window"] else 120
        st.session_state["live_factors"] = st.session_state["live_factors"].iloc[-win:]
        st.session_state["live_portfolio"] = st.session_state["live_portfolio"].iloc[-win:]

    factors_df = st.session_state.get("live_factors")
    portfolio_df = st.session_state.get("live_portfolio")
    if factors_df is not None and portfolio_df is not None:
        st.caption(f"Live buffer: {factors_df.shape[0]} points")
        st.line_chart(portfolio_df["Portfolio"].tail(200))


def _run_analysis(factors: pd.DataFrame, portfolio: pd.DataFrame, rolling_window_in: int | None):
    # Ensure directories
    paths = Paths.from_root(str(APP_ROOT))

    alpha, betas, r2, y_hat = M.fit_ols_excess(
        portfolio=portfolio["Portfolio"], rf=portfolio["RF"], factors=factors
    )
    contrib_pct = M.attribution_percent(
        factors, alpha, betas, (portfolio["Portfolio"] - portfolio["RF"]).to_numpy()
    )

    betas_df = pd.DataFrame({"Beta": [betas[k] for k in betas]}, index=list(betas.keys()))

    # Charts
    betas_png = paths.charts / "betas.png"
    attr_png = paths.charts / "attribution.png"
    cum_png = paths.charts / "cumulative.png"
    V.plot_betas(betas, betas_png)
    V.plot_attribution(contrib_pct, attr_png)
    V.plot_cumulative(portfolio["Portfolio"], portfolio["RF"], y_hat, portfolio.index, cum_png)

    rolling_png = None
    if rolling_window_in and rolling_window_in > 0:
        roll = M.rolling_betas(portfolio["Portfolio"], portfolio["RF"], factors, window=int(rolling_window_in))
        if not roll.empty:
            rolling_png = paths.charts / "rolling_betas_heatmap.png"
            V.plot_rolling_heatmap(roll, rolling_png)
            (paths.output / "tables").mkdir(parents=True, exist_ok=True)
            roll.to_csv(paths.output / "tables" / "rolling_betas.csv")

            roll_alpha = M.rolling_alpha(portfolio["Portfolio"], portfolio["RF"], factors, window=int(rolling_window_in))
            V.plot_rolling_alphas(roll_alpha, paths.charts / "rolling_alpha.png")

    report_path = R.build_html_report(
        paths.output,
        alpha,
        betas_df,
        r2,
        (portfolio["Portfolio"] - portfolio["RF"]).mean(),
        betas_png,
        attr_png,
        cum_png,
        rolling_png,
    )
    R.export_csvs(
        paths.output, alpha, betas_df, r2, (portfolio["Portfolio"] - portfolio["RF"]).mean()
    )
    return alpha, betas_df, r2, report_path, betas_png, attr_png, cum_png, rolling_png


if run_btn:
    if factors_df is None or portfolio_df is None:
        st.warning("Please provide both factors and portfolio data.")
    else:
        try:
            roll_opt = int(rolling_window) if rolling_window and rolling_window > 0 else None
            # In live mode, run automatically each refresh
            alpha, betas_df, r2, report_path, betas_png, attr_png, cum_png, rolling_png = _run_analysis(
                factors_df.copy(), portfolio_df.copy(), roll_opt
            )

            st.success("Analysis complete.")

            mean_excess = (portfolio_df["Portfolio"] - portfolio_df["RF"]).mean()
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Alpha (monthly)", f"{alpha:0.6f}")
            with m2:
                st.metric("R-squared", f"{r2:0.4f}")
            with m3:
                st.metric("Mean Excess Return (monthly)", f"{mean_excess:0.6f}")

            tab_overview, tab_charts, tab_tables, tab_downloads = st.tabs(["Overview", "Charts", "Tables", "Downloads"])

            with tab_overview:
                # Generate detailed explanations
                if mode in ["Real Stock Portfolio", "Upload CSVs", "Generate synthetic"]:
                    try:
                        explanation_generator = ExplanationsGenerator()
                        # Convert betas_df to dict
                        betas_dict = betas_df['Beta'].to_dict()
                        # Generate comprehensive explanation
                        explanation = explanation_generator.generate_overall_explanation(
                            alpha=alpha,
                            betas=betas_dict,
                            r_squared=r2,
                            mean_excess_return=mean_excess,
                            portfolio_returns=portfolio_df["Portfolio"].tolist()
                        )
                        # Display key insights
                        insights = explanation_generator.generate_summary_insights(explanation)
                        st.subheader("📋 Key Insights")
                        for insight in insights:
                            st.markdown(f"• {insight}")
                        # Detailed explanations
                        with st.expander("📖 Detailed Factor Explanations", expanded=False):
                            for factor_exp in explanation.factor_explanations:
                                st.markdown(f"""
                                ### {factor_exp.name} (β = {factor_exp.current_value:.3f})
                                
                                **Description**: {factor_exp.description}
                                
                                **Current Exposure**: {factor_exp.interpretation}
                                
                                **Practical Meaning**: {factor_exp.practical_meaning}
                                
                                **Typical Range**: {factor_exp.typical_range}
                                
                                **Impact Assessment**: {factor_exp.assessment}
                                
                                ---
                                """)
                        # Portfolio profile and recommendations
                        st.subheader("📈 Portfolio Assessment")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            **Portfolio Profile**:  
                            {explanation.portfolio_profile}
                            
                            **Risk Assessment**:  
                            {explanation.risk_assessment}
                            """)
                        with col2:
                            st.markdown("**Recommendations**:")
                            for i, rec in enumerate(explanation.recommendations, 1):
                                st.markdown(f"{i}. {rec}")
                        # Option to download full explanation
                        explanation_text = explanation_generator.format_explanation_text(explanation)
                        st.download_button(
                            "📄 Download Full Explanation Report",
                            data=explanation_text.encode('utf-8'),
                            file_name=f"portfolio_explanation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.warning(f"Could not generate detailed explanations: {e}")

            with tab_charts:
                st.subheader("📊 Factor Exposures (Betas)")
                st.dataframe(betas_df)
                st.image(str(betas_png))
                st.subheader("Attribution")
                st.image(str(attr_png))
                st.subheader("Cumulative Returns")
                st.image(str(cum_png))
                # Rolling charts if available
                charts_dir = Paths.from_root(str(APP_ROOT)).charts
                r2_png = charts_dir / "rolling_r2.png"
                alpha_png = charts_dir / "rolling_alpha.png"
                if r2_png.exists():
                    st.subheader("Rolling R²")
                    st.image(str(r2_png))
                if alpha_png.exists():
                    st.subheader("Rolling Alpha")
                    st.image(str(alpha_png))

            with tab_tables:
                tables_dir = (Paths.from_root(str(APP_ROOT)).output / "tables")
                if tables_dir.exists():
                    csvs = sorted(tables_dir.glob("*.csv"))
                    if csvs:
                        for p in csvs:
                            st.write(p.name)
                            st.dataframe(pd.read_csv(p))
                    else:
                        st.info("No tables generated yet.")
                else:
                    st.info("No tables directory found.")

            with tab_downloads:
                # Report
                with open(report_path, "rb") as fh:
                    st.download_button("Download HTML report", data=fh.read(), file_name="report.html", mime="text/html")
                # Tables (zip in-memory)
                tables_dir = (Paths.from_root(str(APP_ROOT)).output / "tables")
                if tables_dir.exists():
                    buf = io.BytesIO()
                    import zipfile

                    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for p in tables_dir.glob("*.csv"):
                            zf.write(p, arcname=p.name)
                    buf.seek(0)
                    st.download_button("Download tables (zip)", data=buf.read(), file_name="tables.zip")

        except Exception as e:
            st.error(f"Analysis failed: {e}")


