#!/usr/bin/env python3
"""
Setup script for Routesit AI
Installs dependencies and initializes the system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Basic dependencies for the simple app
    basic_deps = [
        "streamlit",
        "pandas",
        "plotly",
        "pyyaml"
    ]
    
    for dep in basic_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data/interventions",
        "data/models",
        "data/embeddings",
        "data/samples",
        "logs",
        "data/exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def check_database():
    """Check if intervention database exists"""
    print("🗄️ Checking intervention database...")
    
    db_file = Path("data/interventions/interventions.json")
    if db_file.exists():
        print(f"✅ Intervention database found: {db_file}")
        return True
    else:
        print(f"❌ Intervention database not found: {db_file}")
        print("Please run the database generation script first:")
        print("python scripts/generate_database.py")
        return False

def main():
    """Main setup function"""
    print("🛣️ Routesit AI Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️ Some dependencies failed to install. You may need to install them manually.")
    
    # Check database
    if not check_database():
        print("⚠️ Database not found. Please generate it first.")
    
    print("\n🎉 Setup completed!")
    print("\nTo run the application:")
    print("streamlit run simple_app.py")
    print("\nOr for the full version:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
