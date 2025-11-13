#!/bin/bash
# Setup script for Linux/Mac

echo "========================================"
echo "Plant-Specific Fertilizer & Soil Recommendation System"
echo "Setup Script for Linux/Mac"
echo "========================================"
echo ""

echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi
python3 --version
echo ""

echo "[2/4] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "Virtual environment created"
fi
echo ""

echo "[3/4] Activating virtual environment..."
source venv/bin/activate
echo ""

echo "[4/4] Installing dependencies..."
echo "This may take several minutes..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Start the backend:"
echo "     source venv/bin/activate"
echo "     uvicorn backend.main:app --reload"
echo ""
echo "  2. Open frontend/index.html in your browser"
echo ""
echo "  3. Or run the quick test:"
echo "     python quick_start.py"
echo ""
echo "========================================"
