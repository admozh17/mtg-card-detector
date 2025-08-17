#!/bin/bash

# MTG Card Detector Startup Script
# This script starts both the Python backend and React Native frontend

echo "🚀 Starting MTG Card Detector..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Start backend in background
echo "🔧 Starting Python backend..."
python3 app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! curl -s http://localhost:5001/health > /dev/null; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Backend started successfully on http://localhost:5001"

# Start frontend
echo "📱 Starting React Native frontend..."
echo "🔗 Backend API: http://192.168.200.171:5001"
echo "📱 Frontend: Expo development server"
echo ""
echo "📋 Instructions:"
echo "1. The backend is now running on http://192.168.200.171:5001"
echo "2. The frontend will start in a new terminal window"
echo "3. Scan the QR code with Expo Go app on your device"
echo "4. Make sure your device and computer are on the same network"
echo ""
echo "🛑 To stop all services, press Ctrl+C in this terminal"
echo ""

# Start frontend in new terminal (macOS/Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd \"'$PWD'\" && npm start"'
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    gnome-terminal -- bash -c "cd \"$PWD\" && npm start; exec bash" 2>/dev/null || \
    xterm -e "cd \"$PWD\" && npm start; exec bash" 2>/dev/null || \
    konsole -e "cd \"$PWD\" && npm start; exec bash" 2>/dev/null || \
    echo "⚠️  Could not open new terminal automatically. Please run 'npm start' in a new terminal."
else
    echo "⚠️  Please run 'npm start' in a new terminal window"
fi

# Wait for user to stop
echo "⏳ Services are running. Press Ctrl+C to stop..."
trap "echo '🛑 Stopping services...'; kill $BACKEND_PID 2>/dev/null; exit 0" INT

# Keep script running
while true; do
    sleep 1
done
