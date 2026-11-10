@echo off
setlocal
set "CARDIO_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%CARDIO_PYTHON%" (
    echo Environment not found. Follow START_HERE.txt in the project folder first.
    pause
    exit /b 1
)
"%CARDIO_PYTHON%" -m streamlit run "%~dp0app.py" --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false --theme.base=light --theme.primaryColor=#176b78 --theme.backgroundColor=#f6f8fa --theme.secondaryBackgroundColor=#ffffff --theme.textColor=#18334a
pause
