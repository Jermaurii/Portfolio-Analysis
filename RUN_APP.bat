@echo off
echo Starting Portfolio Factor Analysis App...
echo ========================================

cd /d "%~dp0"

echo Current directory: 
cd
echo.

echo Checking if streamlit_app.py exists...
if not exist "streamlit_app.py" (
    echo ERROR: streamlit_app.py not found!
    echo Please run this from the Portfolio Factor Analysis directory
    pause
    exit /b 1
)

echo streamlit_app.py found!
echo.

echo Installing/updating required packages...
pip install -r requirements.txt

echo.
echo Starting Streamlit server...
echo.
echo The app will open in your browser automatically
echo If it doesn't open, manually go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================

python -m streamlit run streamlit_app.py --server.port 8501

pause
