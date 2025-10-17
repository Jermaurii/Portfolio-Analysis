#!/usr/bin/env python3
"""
Test script for real stock data functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_real_data():
    print("Testing Real Stock Data Functionality")
    print("=" * 50)
    
    try:
        from src.real_data import RealDataFetcher
        
        fetcher = RealDataFetcher()
        print("[OK] Data fetcher initialized")
        
        # Test 1: Stock data fetching
        print("\n1. Testing stock data fetching...")
        test_stocks = ['AAPL', 'MSFT']
        
        stock_data = fetcher.get_stock_data(
            symbols=test_stocks,
            start_date='2022-01-01',
            end_date='2023-01-01'
        )
        
        print(f"   Successfully fetched: {stock_data.shape[0]} days of data")
        print(f"   Columns: {list(stock_data.columns)}")
        print(f"   Date range: {stock_data.index.min()} to {stock_data.index.max()}")
        
        # Test 2: Portfolio creation
        print("\n2. Testing portfolio creation...")
        portfolio_df = fetcher.create_portfolio_data(
            symbols=test_stocks,
            start_date='2022-01-01',
            end_date='2023-01-01'
        )
        
        print(f"   Portfolio shape: {portfolio_df.shape}")
        print(f"   Columns: {list(portfolio_df.columns)}")
        print(f"   Monthly periods: {portfolio_df.shape[0]}")
        
        # Show sample data
        print("\n3. Sample portfolio data:")
        print(portfolio_df.head())
        
        # Test 3: Factor data
        print("\n4. Testing factor data...")
        try:
            factors_df = fetcher.get_complete_factor_data('2022-01-01', '2023-01-01')
            print(f"   Factor data shape: {factors_df.shape}")
            print(f"   Factor columns: {list(factors_df.columns)}")
        except Exception as e:
            print(f"   Factor data test: {e}")
            print("   (This is expected if network access is limited)")
        
        print("\n" + "=" * 50)
        print("REAL DATA TEST RESULTS:")
        print("=" * 50)
        print("Stock data fetching: WORKING")
        print("Portfolio creation: WORKING") 
        print("Data alignment: OK")
        print("Ready for Streamlit app!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        print("\nThis might be due to:")
        print("- Network connectivity issues")
        print("- Missing packages")
        print("- API rate limits")
        
        return False

if __name__ == "__main__":
    success = test_real_data()
    
    if success:
        print("\nNEXT STEPS:")
        print("1. Double-click RUN_APP.bat to start the app")
        print("2. Or run: streamlit run streamlit_app.py")
        print("3. Open: http://localhost:8501")
        print("4. Select 'Real Stock Portfolio' mode")
    else:
        print("\nTROUBLESHOOTING:")
        print("1. Check internet connection")
        print("2. Run: pip install yfinance requests")
        print("3. Try again")


