#!/bin/bash

# PDF Intelligence Platform - Simple Startup Script
echo "🚀 PDF Intelligence Platform"
echo "============================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if requirements are installed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/lib/python*/site-packages/fastapi" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Please create a .env file with your OpenAI API key"
    echo "   Example: echo 'OPENAI_API_KEY=your_key_here' > .env"
fi

# Start the application
echo "🎯 Starting application..."
echo "   API Docs: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo "   Press Ctrl+C to stop"
echo ""

python3 run.py
