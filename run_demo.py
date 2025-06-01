#!/usr/bin/env python3
"""
Quick demo runner for Text Classification System
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def check_requirements():
    """Check if required tools are installed"""
    required_tools = {
        'docker': 'Docker is required. Please install Docker Desktop.',
        'docker-compose': 'Docker Compose is required. Please install Docker Compose.'
    }
    
    for tool, message in required_tools.items():
        try:
            subprocess.run([tool, '--version'], 
                         capture_output=True, check=True)
            print(f"✅ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {message}")
            return False
    
    return True

def build_and_run():
    """Build and run the demo using Docker Compose"""
    print("🏗️  Building and starting the Text Classification Demo...")
    
    try:
        # Build and start services
        result = subprocess.run([
            'docker-compose', 'up', '--build', '-d'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to start services: {result.stderr}")
            return False
        
        print("✅ Services started successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        return False

def wait_for_services():
    """Wait for services to be ready"""
    print("⏳ Waiting for services to be ready...")
    
    import requests
    import time
    
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Check backend health
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                break
        except requests.exceptions.RequestException:
            pass
        
        attempt += 1
        time.sleep(2)
        print(f"⏳ Attempt {attempt}/{max_attempts}...")
    
    if attempt >= max_attempts:
        print("❌ Services did not start in time. Check logs with: docker-compose logs")
        return False
    
    return True

def open_browser():
    """Open the demo in browser"""
    print("🌐 Opening demo in browser...")
    try:
        webbrowser.open('http://localhost:3000')
        print("✅ Demo opened in browser!")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print("📱 Please manually open: http://localhost:3000")

def show_info():
    """Show demo information"""
    print("\n" + "="*60)
    print("🎉 TEXT CLASSIFICATION DEMO IS RUNNING!")
    print("="*60)
    print("📱 Frontend:     http://localhost:3000")
    print("🔧 Backend API:  http://localhost:8000")
    print("📚 API Docs:     http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("="*60)
    print("\n🧪 Try these examples:")
    print("• Sentiment: 'I love this product!'")
    print("• Spam: 'FREE MONEY! Click now!'")
    print("• Topic: 'The new AI technology is amazing'")
    print("\n⏹️  To stop: docker-compose down")
    print("📋 To view logs: docker-compose logs")
    print("="*60)

def main():
    """Main demo runner"""
    print("🚀 Starting Text Classification Demo...")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install required tools and try again.")
        sys.exit(1)
    
    # Build and run
    if not build_and_run():
        print("\n❌ Failed to start demo. Please check the logs.")
        sys.exit(1)
    
    # Wait for services
    if not wait_for_services():
        print("\n❌ Services did not start properly.")
        sys.exit(1)
    
    # Open browser
    open_browser()
    
    # Show info
    show_info()
    
    print("\n🎯 Demo is ready! Enjoy testing the Text Classification System!")

if __name__ == "__main__":
    main()
