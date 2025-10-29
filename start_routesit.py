#!/usr/bin/env python3
"""
Routesit AI Startup Script
Run this script to start the award-winning road safety system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    try:
        print("Starting Routesit AI...")
        print("=" * 50)
        
        # Import and run the application
        from enhanced_app_v2 import main
        main()
        
    except Exception as e:
        print(f"Error starting Routesit AI: {e}")
        sys.exit(1)
