#!/usr/bin/env python3
"""
Internet Connectivity Manager
Handles internet detection, user preferences, and conditional feature loading
"""

import os
import json
import logging
import requests
import socket
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class InternetSettings:
    """Internet usage settings"""
    use_internet: bool = False
    allow_web_scraping: bool = False
    allow_updates: bool = False
    allow_api_calls: bool = False
    last_checked: Optional[str] = None
    connection_status: str = "unknown"  # unknown, connected, disconnected

class InternetManager:
    """Manages internet connectivity and user preferences"""
    
    def __init__(self, config_path: str = "config/internet_settings.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = InternetSettings()
        self._load_settings()
        
    def _load_settings(self):
        """Load settings from config file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.settings.use_internet = data.get("use_internet", False)
                    self.settings.allow_web_scraping = data.get("allow_web_scraping", False)
                    self.settings.allow_updates = data.get("allow_updates", False)
                    self.settings.allow_api_calls = data.get("allow_api_calls", False)
                    self.settings.last_checked = data.get("last_checked")
                    self.settings.connection_status = data.get("connection_status", "unknown")
            except Exception as e:
                logger.error(f"Failed to load internet settings: {e}")
                self.settings = InternetSettings()
    
    def _save_settings(self):
        """Save settings to config file"""
        try:
            data = {
                "use_internet": self.settings.use_internet,
                "allow_web_scraping": self.settings.allow_web_scraping,
                "allow_updates": self.settings.allow_updates,
                "allow_api_calls": self.settings.allow_api_calls,
                "last_checked": self.settings.last_checked,
                "connection_status": self.settings.connection_status
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save internet settings: {e}")
    
    def check_connectivity(self, timeout: int = 5) -> bool:
        """Check if internet connection is available"""
        try:
            # Try multiple methods for robust detection
            methods = [
                self._check_http_request,
                self._check_dns_resolution,
                self._check_socket_connection
            ]
            
            for method in methods:
                try:
                    if method(timeout):
                        self.settings.connection_status = "connected"
                        return True
                except Exception:
                    continue
            
            self.settings.connection_status = "disconnected"
            return False
            
        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            self.settings.connection_status = "disconnected"
            return False
    
    def _check_http_request(self, timeout: int) -> bool:
        """Check connectivity via HTTP request"""
        try:
            response = requests.get(
                "http://www.google.com", 
                timeout=timeout,
                headers={"User-Agent": "Routesit-AI/1.0"}
            )
            return response.status_code == 200
        except:
            return False
    
    def _check_dns_resolution(self, timeout: int) -> bool:
        """Check connectivity via DNS resolution"""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            return True
        except:
            return False
    
    def _check_socket_connection(self, timeout: int) -> bool:
        """Check connectivity via socket connection"""
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 80))
            return True
        except:
            return False
    
    def get_user_preference(self) -> bool:
        """Get user preference for internet usage"""
        return self.settings.use_internet
    
    def set_preference(self, use_internet: bool, 
                      allow_web_scraping: bool = None,
                      allow_updates: bool = None,
                      allow_api_calls: bool = None):
        """Set user preferences for internet usage"""
        self.settings.use_internet = use_internet
        
        if allow_web_scraping is not None:
            self.settings.allow_web_scraping = allow_web_scraping
        if allow_updates is not None:
            self.settings.allow_updates = allow_updates
        if allow_api_calls is not None:
            self.settings.allow_api_calls = allow_api_calls
        
        self._save_settings()
        logger.info(f"Internet preferences updated: use_internet={use_internet}")
    
    def should_use_internet(self) -> bool:
        """Check if internet should be used based on connectivity and preferences"""
        if not self.settings.use_internet:
            return False
        
        # Check connectivity if not recently checked
        if self.settings.connection_status == "unknown":
            return self.check_connectivity()
        
        return self.settings.connection_status == "connected"
    
    def should_web_scrape(self) -> bool:
        """Check if web scraping should be enabled"""
        return (self.should_use_internet() and 
                self.settings.allow_web_scraping)
    
    def should_check_updates(self) -> bool:
        """Check if updates should be checked"""
        return (self.should_use_internet() and 
                self.settings.allow_updates)
    
    def should_make_api_calls(self) -> bool:
        """Check if API calls should be made"""
        return (self.should_use_internet() and 
                self.settings.allow_api_calls)
    
    def get_status_info(self) -> Dict[str, any]:
        """Get current internet status information"""
        return {
            "is_connected": self.check_connectivity(),
            "user_preference": self.settings.use_internet,
            "allow_web_scraping": self.settings.allow_web_scraping,
            "allow_updates": self.settings.allow_updates,
            "allow_api_calls": self.settings.allow_api_calls,
            "connection_status": self.settings.connection_status,
            "last_checked": self.settings.last_checked
        }
    
    def create_startup_modal_data(self) -> Dict[str, any]:
        """Create data for startup modal"""
        is_connected = self.check_connectivity()
        
        return {
            "internet_detected": is_connected,
            "current_preference": self.settings.use_internet,
            "modal_title": "Internet Connectivity Detected",
            "modal_message": "Routesit AI has detected an internet connection. Would you like to enable web scraping and real-time updates?",
            "options": {
                "enable_all": "Enable all internet features",
                "enable_selective": "Choose specific features",
                "disable": "Work offline only"
            },
            "features": {
                "web_scraping": {
                    "name": "Web Scraping",
                    "description": "Scrape accident news and research papers",
                    "enabled": self.settings.allow_web_scraping
                },
                "updates": {
                    "name": "Real-time Updates", 
                    "description": "Check for latest IRC/MoRTH updates",
                    "enabled": self.settings.allow_updates
                },
                "api_calls": {
                    "name": "API Calls",
                    "description": "Make external API calls for enhanced analysis",
                    "enabled": self.settings.allow_api_calls
                }
            }
        }

def create_internet_modal():
    """Create Streamlit modal for internet preferences"""
    import streamlit as st
    
    manager = InternetManager()
    modal_data = manager.create_startup_modal_data()
    
    if modal_data["internet_detected"] and not manager.get_user_preference():
        with st.container():
            st.markdown("### 🌐 Internet Connectivity Detected")
            st.markdown(modal_data["modal_message"])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Enable All Features", key="enable_all"):
                    manager.set_preference(
                        use_internet=True,
                        allow_web_scraping=True,
                        allow_updates=True,
                        allow_api_calls=True
                    )
                    st.success("All internet features enabled!")
                    st.rerun()
            
            with col2:
                if st.button("⚙️ Choose Features", key="enable_selective"):
                    st.session_state.show_internet_settings = True
            
            with col3:
                if st.button("🚫 Offline Only", key="disable"):
                    manager.set_preference(use_internet=False)
                    st.info("Working in offline mode")
                    st.rerun()
            
            # Show detailed settings if requested
            if st.session_state.get("show_internet_settings", False):
                st.markdown("### Internet Feature Settings")
                
                web_scraping = st.checkbox(
                    "Web Scraping", 
                    value=modal_data["features"]["web_scraping"]["enabled"],
                    help=modal_data["features"]["web_scraping"]["description"]
                )
                
                updates = st.checkbox(
                    "Real-time Updates",
                    value=modal_data["features"]["updates"]["enabled"], 
                    help=modal_data["features"]["updates"]["description"]
                )
                
                api_calls = st.checkbox(
                    "API Calls",
                    value=modal_data["features"]["api_calls"]["enabled"],
                    help=modal_data["features"]["api_calls"]["description"]
                )
                
                if st.button("Save Settings"):
                    manager.set_preference(
                        use_internet=True,
                        allow_web_scraping=web_scraping,
                        allow_updates=updates,
                        allow_api_calls=api_calls
                    )
                    st.success("Settings saved!")
                    st.session_state.show_internet_settings = False
                    st.rerun()

def main():
    """Test the Internet Manager"""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Internet Manager...")
    
    manager = InternetManager()
    
    # Check connectivity
    is_connected = manager.check_connectivity()
    print(f"Internet connected: {is_connected}")
    
    # Get status
    status = manager.get_status_info()
    print(f"Status: {status}")
    
    # Test preferences
    print(f"Current preference: {manager.get_user_preference()}")
    
    # Set test preference
    manager.set_preference(use_internet=True, allow_web_scraping=True)
    print(f"Updated preference: {manager.get_user_preference()}")
    
    # Test conditional features
    print(f"Should web scrape: {manager.should_web_scrape()}")
    print(f"Should check updates: {manager.should_check_updates()}")

if __name__ == "__main__":
    main()
