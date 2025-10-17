#!/usr/bin/env python3
"""
Real-Life Portfolio Factor Analysis Demo
Demonstrates the portfolio analysis capabilities without needing Streamlit
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.real_data import RealDataFetcher
    from src.explanations import ExplanationsGenerator
    from src.model import fit_ols_excess
    print("[OK] Successfully imported portfolio analysis modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Using demo mode...")

def demo_portfolio_analysis():
    """Run a demonstration of real portfolio analysis"""
    print("\n" + "="*60)
    print("REAL-LIFE PORTFOLIO FACTOR ANALYSIS DEMO")
    print("="*60)
    
    try:
        # Initialize data fetcher
        print("\n🔍 Initializing data fetcher...")
        fetcher = RealDataFetcher()
        
        # Show available stock sectors
        print("\n🏢 Available stock sectors:")
        popular_stocks = fetcher.get_popular_stocks()
        for sector, stocks in popular_stocks.items():
            print(f"  {sector}: {', '.join(stocks[:4])}...")
        
        # Demonstrate Technology portfolio analysis
        print("\n💻 Analyzing Technology Portfolio:")
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
        
        print(f"📊 Fetching data for: {', '.join(tech_stocks)}")
        print("   This may take a few moments...")
        
        # Get portfolio data (shorter period for demo)
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 years
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Create portfolio
        portfolio_df = fetcher.create_portfolio_data(
            symbols=tech_stocks,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get factors
        factors_df = fetcher.get_complete_factor_data(start_date, end_date)
        
        # Align data
        common_index = portfolio_df.index.intersection(factors_df.index)
        portfolio_aligned = portfolio_df.loc[common_index]
        factors_aligned = factors_df.loc[common_index]
        
        print(f"✅ Loaded {len(portfolio_aligned)} months of data")
        
        # Run analysis
        print("\n🧮 Running factor analysis...")
        alpha, betas, r2, y_hat = fit_ols_excess(
            portfolio=portfolio_aligned["Portfolio"],
            rf=portfolio_aligned["RF"],
            factors=factors_aligned
        )
        
        # Display results
        print("\n📊 ANALYSIS RESULTS:")
        print("-" * 40)
        print(f"Alpha (monthly):        {alpha:.6f} ({alpha*12*100:.2f}% annualized)")
        print(f"R-squared:             {r2:.4f} ({r2*100:.1f}% variance explained)")
        
        mean_excess = (portfolio_aligned["Portfolio"] - portfolio_aligned["RF"]).mean()
        print(f"Mean Excess Return:    {mean_excess:.6f} ({mean_excess*12*100:.2f}% annualized)")
        
        print("\n📈 FACTOR EXPOSURES:")
        print("-" * 40)
        for factor, beta in betas.items():
            factor_names = {
                'MKT_RF': 'Market Risk Premium',
                'SMB': 'Size Factor (Small-Cap)',
                'HML': 'Value Factor',
                'MOM': 'Momentum Factor',
                'RMW': 'Profitability Factor',
                'CMA': 'Investment Factor'
            }
            name = factor_names.get(factor, factor)
            print(f"{name:.<25} {beta:>8.4f}")
        
        # Generate explanations
        print("\n🧠 EXPLANATION:")
        print("-" * 40)
        try:
            explanation_generator = ExplanationsGenerator()
            
            # Convert betas_df to dict format
            betas_dict = betas
            
            # Generate explanation
            explanation = explanation_generator.generate_overall_explanation(
                alpha=alpha,
                betas=betas_dict,
                r_squared=r2,
                mean_excess_return=mean_excess
            )
            
            # Show key insights
            insights = explanation_generator.generate_summary_insights(explanation)
            for insight in insights:
                print(f"• {insight}")
            
            # Show portfolio profile
            print(f"\n📊 Portfolio Profile: {explanation.portfolio_profile}")
            print(f"⚠️  Risk Assessment: {explanation.risk_assessment}")
            
            # Show recommendations
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(explanation.recommendations, 1):
                print(f"   {i}. {rec}")
                
        except Exception as e:
            print(f"❌ Could not generate explanations: {e}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 To use the full interactive interface:")
        print("   1. Fix Streamlit connectivity issues")
        print("   2. Or run: python -m streamlit run streamlit_app.py")
        print("   3. Open http://localhost:8501 in your browser")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("\n🔧 Falling back to synthetic demo...")
        synthetic_demo()

def synthetic_demo():
    """Fallback synthetic demo if real data fails"""
    print("\n📊 Running synthetic portfolio analysis demo...")
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq='M')
    
    # Synthetic factors
    factors_data = {
        'MKT_RF': np.random.normal(0.005, 0.04, 60),
        'SMB': np.random.normal(0.002, 0.03, 60),
        'HML': np.random.normal(-0.001, 0.025, 60),
        'MOM': np.random.normal(0.004, 0.035, 60),
        'RMW': np.random.normal(0.0025, 0.03, 60),
        'CMA': np.random.normal(0.002, 0.03, 60),
        'RF': np.random.uniform(0.0015, 0.004, 60)
    }
    
    factors_df = pd.DataFrame(factors_data, index=dates)
    factors_df.index.name = 'date'
    
    # Synthetic portfolio
    alpha = 0.0008
    betas_true = {'MKT_RF': 1.05, 'SMB': 0.25, 'HML': -0.15, 'MOM': 0.40, 'RMW': 0.10, 'CMA': -0.05}
    eps = np.random.normal(0.0, 0.01, 60)
    
    portfolio_returns = []
    rf_series = factors_df['RF'].values
    
    for i in range(len(factors_df)):
        excess = alpha + sum(betas_true[k] * factors_df.iloc[i][k] for k in betas_true) + eps[i]
        total = excess + rf_series[i]
        portfolio_returns.append(total)
    
    portfolio_df = pd.DataFrame({
        'Portfolio': portfolio_returns,
        'RF': rf_series
    }, index=dates)
    portfolio_df.index.name = 'date'
    
    # Run analysis
    alpha_est, betas_est, r2, y_hat = fit_ols_excess(
        portfolio=portfolio_df["Portfolio"],
        rf=portfolio_df["RF"],
        factors=factors_df
    )
    
    print(f"\n📊 SYNTHETIC PORTFOLIO ANALYSIS:")
    print("-" * 40)
    print(f"True Alpha:    {alpha:.6f}")
    print(f"Estimated Alpha: {alpha_est:.6f}")
    print(f"R-squared:    {r2:.4f}")
    
    print(f"\n📈 Factor Exposures (Established vs True):")
    print("-" * 40)
    for factor in ['MKT_RF', 'SMB', 'HML', 'MOM', 'RMW', 'CMA']:
        true_val = betas_true[factor]
        est_val = betas_est.get(factor, 0)
        print(f"{factor:10}    True: {true_val:>8.4f}    Est: {est_val:>8.4f}")
    
    print(f"\n✅ Synthetic demo completed!")
    print("This shows how the factor analysis works with controlled data.")

if __name__ == "__main__":
    print("🚀 Starting Portfolio Factor Analysis Demo...")
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("⚠️  Warning: streamlit_app.py not found in current directory")
        print("   Please run this from the project root directory")
    
    demo_portfolio_analysis()
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("="*60)
    print("1. Fix Streamlit server startup")
    print("2. Run: python -m streamlit run streamlit_app.py")
    print("3. Open: http://localhost:8501")
    print("4. Select 'Real Stock Portfolio' mode")
    print("5. Choose stocks and analyze!")
    print("="*60)
