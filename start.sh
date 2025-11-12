#!/bin/bash

# AI Piano Teacher - Docker Startup Script
# This script starts the complete AI Piano Teacher system with Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1 && ! docker compose version > /dev/null 2>&1; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p scores uploads temp logs
    print_success "Directories created"
}

# Function to check for environment file
check_environment() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f env.example ]; then
            cp env.example .env
            print_success "Created .env file from template"
            print_warning "Please review and update the .env file with your settings"
        else
            print_warning "No env.example found. Using default environment variables."
        fi
    else
        print_success ".env file found"
    fi
}

# Function to build and start services
start_services() {
    print_status "Building and starting AI Piano Teacher services..."
    
    # Use docker compose if available, otherwise fall back to docker-compose
    if docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    print_status "Building images (this may take a few minutes)..."
    $COMPOSE_CMD build
    
    print_status "Starting services..."
    $COMPOSE_CMD up -d
    
    print_success "Services started successfully!"
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for database..."
    for i in {1..30}; do
        if docker-compose exec -T db pg_isready -U piano_user -d piano_analysis > /dev/null 2>&1; then
            print_success "Database is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Database failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Wait for backend API
    print_status "Waiting for backend API..."
    for i in {1..60}; do
        if curl -f http://localhost:8001/health > /dev/null 2>&1; then
            print_success "Backend API is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            print_error "Backend API failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Wait for frontend
    print_status "Waiting for frontend..."
    for i in {1..60}; do
        if curl -f http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            print_error "Frontend failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
}

# Function to run API tests
run_tests() {
    print_status "Running API tests..."
    
    if [ -f test_apis.py ]; then
        if command -v python3 > /dev/null 2>&1; then
            python3 test_apis.py --url http://localhost:8001
        elif command -v python > /dev/null 2>&1; then
            python test_apis.py --url http://localhost:8001
        else
            print_warning "Python not found. Skipping API tests."
            return
        fi
    else
        print_warning "test_apis.py not found. Skipping API tests."
    fi
}

# Function to display service URLs
show_urls() {
    echo ""
    echo "🎹 AI Piano Teacher is now running!"
    echo ""
    echo "📱 Frontend (Web App):     http://localhost:3000"
    echo "🔧 Backend API:           http://localhost:8001"
    echo "📚 API Documentation:     http://localhost:8001/docs"
    echo "🗄️  Database:              localhost:5432"
    echo "🔄 Redis:                 localhost:6379"
    echo ""
    echo "🔍 To view logs:"
    echo "   docker-compose logs -f [service_name]"
    echo ""
    echo "🛑 To stop all services:"
    echo "   docker-compose down"
    echo ""
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
}

# Main execution
main() {
    echo "🎹 AI Piano Teacher - Docker Startup"
    echo "===================================="
    echo ""
    
    # Pre-flight checks
    check_docker
    check_docker_compose
    
    # Setup
    create_directories
    check_environment
    
    # Start services
    start_services
    
    # Wait for readiness
    wait_for_services
    
    # Show status
    show_status
    
    # Run tests (optional)
    if [ "$1" = "--test" ]; then
        run_tests
    fi
    
    # Display URLs
    show_urls
    
    print_success "AI Piano Teacher is ready to use!"
}

# Handle script arguments
case "$1" in
    --help|-h)
        echo "AI Piano Teacher Startup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --test    Run API tests after startup"
        echo "  --help    Show this help message"
        echo ""
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac