#!/bin/bash

# Multi-Agent LLM Trading System Startup Script

echo "======================================"
echo "Multi-Agent LLM Trading System"
echo "======================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys:"
    echo "  cp .env.example .env"
    echo "  # Edit .env with your API keys"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Check if Redis is running (optional)
if ! command -v redis-cli &> /dev/null; then
    echo "⚠️  Redis not found. Install Redis for full functionality:"
    echo "   Ubuntu/Debian: sudo apt-get install redis-server"
    echo "   macOS: brew install redis"
    echo "   Or use Docker: docker run -d -p 6379:6379 redis:alpine"
fi

# Start the trading system
echo "🚀 Starting trading system..."
python main.py