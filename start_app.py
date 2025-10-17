#!/usr/bin/env python3
"""
Simple launcher script for Portfolio Factor Analysis
"""

import subprocess
import webbrowser
import time
import sys
from pathlib import Path

def start_streamlit_app():
    """Start the Streamlit app and open browser"""
    
    print("Starting Portfolio Factor Analysis App...")
    print("=" * 50)
    
    # Check if we're in the right directory
    app_file = Path("streamlit_app.py")
    if not app_file.exists():
        print("ERROR: streamlit_app.py not found!")
        print("Please run this script from the project root directory")
        return False
    
    # Start Streamlit
    try:
        print("Launching Streamlit server...")
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
        
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("Waiting for server to start...")
        time.sleep(8)  # Give server time to start
        
        # Open browser
        url = "http://localhost:8501"
        print(f"Opening browser: {url}")
        webbrowser.open(url)
        
        print("=" * 50)
        print("SUCCESS! Your app is now running!")
        print("=" * 50)
        print("The browser should open automatically")
        print("If not, manually open: http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            process.terminate()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to start app: {e}")
        return False

if __name__ == "__main__":
    success = start_streamlit_app()
    if not success:
        print("\nAlternative access methods:")
        print("1. Run: python -m streamlit run streamlit_app.py")
        print("2. Then open: http://localhost:8501")
        print("3. Or open: simple_frontend.html")

