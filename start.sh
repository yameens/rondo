#!/bin/bash

# Piano Performance Analysis MVP - Main Startup Script

echo "🎹 Piano Performance Analysis MVP"
echo "=================================="
echo ""

# Function to start backend
start_backend() {
    echo "🚀 Starting Backend API..."
    cd api
    ./start.sh &
    BACKEND_PID=$!
    cd ..
    echo "Backend started with PID: $BACKEND_PID"
    echo ""
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting Frontend..."
    cd web
    ./start.sh &
    FRONTEND_PID=$!
    cd ..
    echo "Frontend started with PID: $FRONTEND_PID"
    echo ""
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "Frontend stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check command line arguments
case "${1:-both}" in
    "backend"|"api")
        start_backend
        echo "Backend running at: http://localhost:8000"
        echo "API docs at: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop"
        wait $BACKEND_PID
        ;;
    "frontend"|"web")
        start_frontend
        echo "Frontend running at: http://localhost:3000"
        echo ""
        echo "Press Ctrl+C to stop"
        wait $FRONTEND_PID
        ;;
    "both"|"all")
        start_backend
        sleep 3  # Give backend time to start
        start_frontend
        echo ""
        echo "🎉 Both services started!"
        echo "   Frontend: http://localhost:3000"
        echo "   Backend:  http://localhost:8000"
        echo "   API Docs: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop all services"
        wait
        ;;
    *)
        echo "Usage: $0 [backend|frontend|both]"
        echo ""
        echo "Options:"
        echo "  backend   - Start only the FastAPI backend"
        echo "  frontend  - Start only the Next.js frontend"
        echo "  both      - Start both services (default)"
        echo ""
        exit 1
        ;;
esac
