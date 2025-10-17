#!/usr/bin/env python3
"""
Direct launcher for Portfolio Factor Analysis
This bypasses Streamlit command issues
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def launch_app():
    print("Portfolio Factor Analysis Launcher")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Check if streamlit_app.py exists
    app_file = Path("streamlit_app.py")
    if not app_file.exists():
        print("ERROR: streamlit_app.py not found!")
        return False
    
    print("streamlit_app.py found!")
    
    # Try multiple approaches to start Streamlit
    approaches = [
        # Method 1: Direct python -m streamlit
        {
            "name": "Method 1: Python module",
            "cmd": [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
        },
        # Method 2: Try streamlit command directly
        {
            "name": "Method 2: Streamlit command", 
            "cmd": ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
        },
        # Method 3: Headless mode
        {
            "name": "Method 3: Headless mode",
            "cmd": [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.port", "8501"]
        }
    ]
    
    for approach in approaches:
        print(f"\nTrying {approach['name']}...")
        try:
            print(f"Command: {' '.join(approach['cmd'])}")
            
            # Start process
            process = subprocess.Popen(
                approach['cmd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment
            time.sleep(5)
            
            # Check if process is still running
            if process.poll() is None:
                print("SUCCESS: Process is running!")
                
                # Try to open browser
                url = "http://localhost:8501"
                print(f"Opening browser: {url}")
                webbrowser.open(url)
                
                print("\n" + "=" * 40)
                print("SUCCESS! App is launching...")
                print("=" * 40)
                print("If browser doesn't open automatically:")
                print("Manually open: http://localhost:8501")
                print("\nPress Ctrl+C to stop the server")
                
                # Keep the process running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    process.terminate()
                
                return True
            else:
                # Process ended, check output
                stdout, stderr = process.communicate()
                print(f"Process ended with return code: {process.returncode}")
                if stdout:
                    print(f"STDOUT: {stdout}")
                if stderr:
                    print(f"STDERR: {stderr}")
                    
        except Exception as e:
            print(f"Method failed: {e}")
            continue
    
    print("\n" + "=" * 40)
    print("All methods failed!")
    print("=" * 40)
    
    # Manual instructions
    print("\nMANUAL INSTRUCTIONS:")
    print("1. Open Command Prompt or PowerShell")
    print("2. Navigate to:", script_dir)
    print("3. Run: python -m streamlit run streamlit_app.py")
    print("4. Open browser: http://localhost:8501")
    
    return False

if __name__ == "__main__":
    success = launch_app()
    
    if not success:
        print("\n" + "=" * 40)
        print("ALTERNATIVE ACCESS")
        print("=" * 40)
        print("Your portfolio analysis demo is working!")
        print("Run: python simple_demo.py")
        print("This shows the core functionality.")
        
        input("\nPress Enter to exit...")
