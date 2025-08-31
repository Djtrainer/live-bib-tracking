#!/bin/bash

# Development script to run frontend and backend separately
echo "Starting Live Bib Tracking in development mode..."
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Shutting down development servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Check if we're in the right directory
if [ ! -d "src/frontend" ] || [ ! -d "src/api_backend" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Start the FastAPI backend
echo "Starting FastAPI backend on http://localhost:8000..."
cd src/api_backend
uvicorn local_server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ../..

# Wait a moment for backend to start
sleep 2

# Start the Vite frontend development server
echo "Starting Vite frontend development server on http://localhost:8080..."
cd src/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "🚀 Development servers started!"
echo ""
echo "Frontend: http://localhost:8080"
echo "Backend API: http://localhost:8000/api"
echo "Backend Docs: http://localhost:8000/docs"
echo ""
echo "The Vite dev server will proxy API requests to the backend."
echo "Press Ctrl+C to stop both servers."
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
