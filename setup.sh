#!/bin/bash

# Face Recognition Attendance System - Ubuntu Setup Script

echo "=================================="
echo "Face Recognition Attendance System"
echo "Ubuntu Setup Script"
echo "=================================="

# Update package list
echo ""
echo "[1/5] Updating package list..."
sudo apt-get update

# Install system dependencies for dlib and face_recognition
echo ""
echo "[2/5] Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libgtk-3-dev \
    libdlib-dev \
    python3-dev \
    python3-pip \
    python3-venv

# Create virtual environment
echo ""
echo "[3/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "[4/5] Activating virtual environment and installing Python packages..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

echo ""
echo "[5/5] Setup complete!"
echo ""
echo "=================================="
echo "To run the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the application:"
echo "     streamlit run app.py"
echo "=================================="
