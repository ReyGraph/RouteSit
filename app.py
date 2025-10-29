#!/usr/bin/env python3
"""
Routesit AI - Road Safety Intervention Decision Intelligence System
Main application entry point

National Road Safety Hackathon 2025 - IIT Madras
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main entry point for Routesit AI"""
    print("🛣️ Routesit AI - Road Safety Intervention Decision Intelligence System")
    print("National Road Safety Hackathon 2025 - IIT Madras")
    print("=" * 70)
    
    try:
        # Import and run the Streamlit app
        from src.interface.web_app import main as run_app
        run_app()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
