#!/bin/bash
"""
Quick Start Script for Enhanced Agentic System Server

This script provides easy ways to run your enhanced agentic system.
"""

echo "🚀 Enhanced Agentic System - Quick Start"
echo "========================================"
echo ""
echo "Available options:"
echo "1. Web Server (recommended)"
echo "2. Production Example"
echo "3. Original Main Script"
echo ""

read -p "Choose an option (1-3): " choice

case $choice in
    1)
        echo "🌐 Starting Web Server..."
        echo "The server will be available at: http://127.0.0.1:5000"
        echo "Press Ctrl+C to stop the server"
        echo ""
        /Users/apple/Downloads/agentic_system_working/workenv/bin/python server.py
        ;;
    2)
        echo "📊 Running Production Example..."
        /Users/apple/Downloads/agentic_system_working/workenv/bin/python production_example.py
        ;;
    3)
        echo "⚙️ Running Original Main Script..."
        /Users/apple/Downloads/agentic_system_working/workenv/bin/python main.py
        ;;
    *)
        echo "❌ Invalid option. Please choose 1, 2, or 3."
        ;;
esac
