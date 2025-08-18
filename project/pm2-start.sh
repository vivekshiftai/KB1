#!/bin/bash

# PM2 Startup Script for PDF Intelligence Platform
# This script helps manage the PM2 process for the FastAPI application

echo "PDF Intelligence Platform - PM2 Management Script"
echo "=================================================="

# Function to start the application
start_app() {
    echo "Starting PDF Intelligence Platform with PM2..."
    
    # Stop any existing processes
    pm2 stop pdf-intelligence-platform 2>/dev/null || true
    pm2 delete pdf-intelligence-platform 2>/dev/null || true
    
    # Start with the startup script configuration
    pm2 start ecosystem-start.config.js
    
    echo "Application started!"
    echo "Check status with: pm2 status"
    echo "View logs with: pm2 logs pdf-intelligence-platform"
}

# Function to stop the application
stop_app() {
    echo "Stopping PDF Intelligence Platform..."
    pm2 stop pdf-intelligence-platform
    echo "Application stopped!"
}

# Function to restart the application
restart_app() {
    echo "Restarting PDF Intelligence Platform..."
    pm2 restart pdf-intelligence-platform
    echo "Application restarted!"
}

# Function to show status
show_status() {
    echo "Current PM2 Status:"
    pm2 status
}

# Function to show logs
show_logs() {
    echo "Recent logs:"
    pm2 logs pdf-intelligence-platform --lines 50
}

# Main script logic
case "$1" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        restart_app
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the application"
        echo "  stop    - Stop the application"
        echo "  restart - Restart the application"
        echo "  status  - Show PM2 status"
        echo "  logs    - Show recent logs"
        exit 1
        ;;
esac
