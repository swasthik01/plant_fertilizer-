@echo off
REM Setup script for Windows
echo ========================================
echo Plant-Specific Fertilizer & Soil Recommendation System
echo Setup Script for Windows
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)
echo.

echo [2/4] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

echo [3/4] Activating virtual environment...
call venv\Scripts\activate
echo.

echo [4/4] Installing dependencies...
echo This may take several minutes...
pip install --upgrade pip
pip install -r requirements.txt
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo   1. Start the backend:
echo      venv\Scripts\activate
echo      uvicorn backend.main:app --reload
echo.
echo   2. Open frontend\index.html in your browser
echo.
echo   3. Or run the quick test:
echo      python quick_start.py
echo.
echo ========================================
pause
