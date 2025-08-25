#!/bin/bash

# Rondo Startup Script
# This script starts the Rondo application in development mode

set -e

echo "🎵 Starting Rondo - Classical Performance Analysis Tool"
echo "=================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if required files exist
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found. Please run this script from the Rondo root directory."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p logs

# Start database and Redis
echo "🗄️  Starting PostgreSQL and Redis..."
docker-compose up -d postgres redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Check if database is ready
until docker-compose exec -T postgres pg_isready -U rondo -d rondo > /dev/null 2>&1; do
    echo "⏳ Still waiting for database..."
    sleep 2
done

echo "✅ Database is ready!"

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
npm install
cd ..

# Start backend
echo "🚀 Starting FastAPI backend..."
cd backend
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 &
BACKEND_PID=$!
cd ..

# Start frontend
echo "🌐 Starting Next.js frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Start Celery worker
echo "⚙️  Starting Celery worker..."
cd backend
celery -A app.worker worker --loglevel=info &
WORKER_PID=$!
cd ..

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Rondo..."
    kill $BACKEND_PID $FRONTEND_PID $WORKER_PID 2>/dev/null || true
    docker-compose down
    echo "✅ Rondo stopped successfully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo ""
echo "🎉 Rondo is starting up!"
echo "=================================================="
echo "📊 Backend API:     http://localhost:8000"
echo "📖 API Docs:        http://localhost:8000/docs"
echo "🌐 Frontend:        http://localhost:3000"
echo "🗄️  Database:        localhost:5432"
echo "📨 Redis:           localhost:6379"
echo ""
echo "⏳ Please wait a moment for all services to start..."
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for services to be ready
sleep 5

# Check if services are running
echo "🔍 Checking service status..."

# Check backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is running"
else
    echo "❌ Backend API is not responding"
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is running"
else
    echo "⏳ Frontend is still starting..."
fi

echo ""
echo "🎵 Rondo is ready! Open http://localhost:3000 in your browser"
echo ""

# Keep script running
wait
