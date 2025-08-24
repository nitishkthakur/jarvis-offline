#!/bin/bash

# ===========================================
# JARVIS AI CHAT SERVER - STARTUP SCRIPT
# ===========================================

echo "🤖 Starting Jarvis AI Chat Server..."
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the chat-ui directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: main.py not found. Please run this script from the chat-ui directory.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ] && [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    source venv/bin/activate
else
    if [ -d "../venv" ]; then
        echo -e "${BLUE}Activating virtual environment...${NC}"
        source ../venv/bin/activate
    else
        source venv/bin/activate
    fi
fi

# Install requirements
echo -e "${BLUE}Installing requirements...${NC}"
pip install -r requirements.txt

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY environment variable is not set.${NC}"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    echo ""
fi

# Start the server
echo -e "${GREEN}Starting FastAPI server...${NC}"
echo ""

# Set default environment variables if not set
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export DEBUG=${DEBUG:-"true"}

# Start the server
python main.py
