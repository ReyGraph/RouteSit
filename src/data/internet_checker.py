#!/usr/bin/env python3
"""
Internet Connectivity Checker for Routesit AI
Detects internet connectivity and manages offline/online modes
"""

import asyncio
import aiohttp
import requests
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class InternetChecker:
    """Check and manage internet connectivity"""
    
    def __init__(self):
        self.test_urls = [
            "https://www.google.com",
            "https://www.timesofindia.indiatimes.com",
            "https://www.hindustantimes.com",
            "https://morth.nic.in",
            "https://www.who.int"
        ]
        
        self.last_check = None
        self.last_status = None
        self.check_interval = 300  # 5 minutes
        self.timeout = 10  # seconds
        
        # Cache file for offline status
        self.cache_file = Path("data/internet_status.json")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def check_connectivity_async(self) -> bool:
        """Asynchronous internet connectivity check"""
        try:
            logger.debug("Checking internet connectivity (async)...")
            
            # Try multiple URLs concurrently
            tasks = []
            for url in self.test_urls[:3]:  # Test first 3 URLs
                task = self._test_url_async(url)
                tasks.append(task)
            
            # Wait for first successful response
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, bool) and result:
                    self.last_check = datetime.now()
                    self.last_status = True
                    self._save_status()
                    logger.info("Internet connectivity confirmed (async)")
                    return True
            
            # No successful connections
            self.last_check = datetime.now()
            self.last_status = False
            self._save_status()
            logger.warning("No internet connectivity detected (async)")
            return False
            
        except Exception as e:
            logger.error(f"Error checking connectivity (async): {e}")
            self.last_check = datetime.now()
            self.last_status = False
            self._save_status()
            return False
    
    def check_connectivity_sync(self) -> bool:
        """Synchronous internet connectivity check"""
        try:
            logger.debug("Checking internet connectivity (sync)...")
            
            for url in self.test_urls[:3]:  # Test first 3 URLs
                if self._test_url_sync(url):
                    self.last_check = datetime.now()
                    self.last_status = True
                    self._save_status()
                    logger.info("Internet connectivity confirmed (sync)")
                    return True
            
            # No successful connections
            self.last_check = datetime.now()
            self.last_status = False
            self._save_status()
            logger.warning("No internet connectivity detected (sync)")
            return False
            
        except Exception as e:
            logger.error(f"Error checking connectivity (sync): {e}")
            self.last_check = datetime.now()
            self.last_status = False
            self._save_status()
            return False
    
    async def _test_url_async(self, url: str) -> bool:
        """Test a single URL asynchronously"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def _test_url_sync(self, url: str) -> bool:
        """Test a single URL synchronously"""
        try:
            response = requests.get(url, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False
    
    def is_connectivity_stale(self) -> bool:
        """Check if connectivity status is stale"""
        if self.last_check is None:
            return True
        
        time_since_check = datetime.now() - self.last_check
        return time_since_check.total_seconds() > self.check_interval
    
    def get_cached_status(self) -> Optional[bool]:
        """Get cached connectivity status"""
        if self.is_connectivity_stale():
            return None
        
        return self.last_status
    
    def _save_status(self):
        """Save connectivity status to cache"""
        try:
            status_data = {
                "last_check": self.last_check.isoformat() if self.last_check else None,
                "last_status": self.last_status,
                "check_interval": self.check_interval,
                "test_urls": self.test_urls
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving connectivity status: {e}")
    
    def _load_status(self):
        """Load connectivity status from cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    status_data = json.load(f)
                
                if status_data.get("last_check"):
                    self.last_check = datetime.fromisoformat(status_data["last_check"])
                    self.last_status = status_data.get("last_status")
                    self.check_interval = status_data.get("check_interval", self.check_interval)
                    
        except Exception as e:
            logger.error(f"Error loading connectivity status: {e}")
    
    def get_connectivity_info(self) -> Dict[str, Any]:
        """Get detailed connectivity information"""
        return {
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_status": self.last_status,
            "is_stale": self.is_connectivity_stale(),
            "check_interval": self.check_interval,
            "test_urls": self.test_urls,
            "cache_file": str(self.cache_file)
        }
    
    def force_check(self) -> bool:
        """Force a connectivity check regardless of cache"""
        logger.info("Forcing connectivity check...")
        return self.check_connectivity_sync()
    
    async def force_check_async(self) -> bool:
        """Force an async connectivity check regardless of cache"""
        logger.info("Forcing async connectivity check...")
        return await self.check_connectivity_async()

class ConnectivityManager:
    """Manage connectivity for different components"""
    
    def __init__(self):
        self.checker = InternetChecker()
        self.checker._load_status()
        
        # Component requirements
        self.component_requirements = {
            "web_scraping": {
                "required": True,
                "fallback": "offline_datasets",
                "retry_interval": 300  # 5 minutes
            },
            "model_download": {
                "required": True,
                "fallback": "local_models",
                "retry_interval": 600  # 10 minutes
            },
            "api_calls": {
                "required": False,
                "fallback": "cached_data",
                "retry_interval": 60  # 1 minute
            },
            "real_time_data": {
                "required": False,
                "fallback": "historical_data",
                "retry_interval": 120  # 2 minutes
            }
        }
        
        # Component status
        self.component_status = {}
    
    def check_component_connectivity(self, component: str) -> Dict[str, Any]:
        """Check connectivity for a specific component"""
        if component not in self.component_requirements:
            return {
                "status": "unknown",
                "message": f"Unknown component: {component}",
                "can_proceed": False
            }
        
        requirements = self.component_requirements[component]
        
        # Check if we have cached status
        cached_status = self.checker.get_cached_status()
        
        if cached_status is not None and not self.checker.is_connectivity_stale():
            # Use cached status
            status = cached_status
        else:
            # Perform fresh check
            status = self.checker.check_connectivity_sync()
        
        # Update component status
        self.component_status[component] = {
            "last_check": datetime.now().isoformat(),
            "status": status,
            "requirements": requirements
        }
        
        if status:
            return {
                "status": "online",
                "message": "Internet connectivity available",
                "can_proceed": True,
                "fallback": None
            }
        else:
            if requirements["required"]:
                return {
                    "status": "offline",
                    "message": f"Internet required for {component}, using fallback: {requirements['fallback']}",
                    "can_proceed": True,
                    "fallback": requirements["fallback"]
                }
            else:
                return {
                    "status": "offline",
                    "message": f"Internet not available for {component}, using fallback: {requirements['fallback']}",
                    "can_proceed": True,
                    "fallback": requirements["fallback"]
                }
    
    async def check_component_connectivity_async(self, component: str) -> Dict[str, Any]:
        """Async check connectivity for a specific component"""
        if component not in self.component_requirements:
            return {
                "status": "unknown",
                "message": f"Unknown component: {component}",
                "can_proceed": False
            }
        
        requirements = self.component_requirements[component]
        
        # Check if we have cached status
        cached_status = self.checker.get_cached_status()
        
        if cached_status is not None and not self.checker.is_connectivity_stale():
            # Use cached status
            status = cached_status
        else:
            # Perform fresh async check
            status = await self.checker.check_connectivity_async()
        
        # Update component status
        self.component_status[component] = {
            "last_check": datetime.now().isoformat(),
            "status": status,
            "requirements": requirements
        }
        
        if status:
            return {
                "status": "online",
                "message": "Internet connectivity available",
                "can_proceed": True,
                "fallback": None
            }
        else:
            if requirements["required"]:
                return {
                    "status": "offline",
                    "message": f"Internet required for {component}, using fallback: {requirements['fallback']}",
                    "can_proceed": True,
                    "fallback": requirements["fallback"]
                }
            else:
                return {
                    "status": "offline",
                    "message": f"Internet not available for {component}, using fallback: {requirements['fallback']}",
                    "can_proceed": True,
                    "fallback": requirements["fallback"]
                }
    
    def get_all_component_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        status = {}
        
        for component in self.component_requirements.keys():
            status[component] = self.check_component_connectivity(component)
        
        return status
    
    def should_retry(self, component: str) -> bool:
        """Check if component should retry connectivity check"""
        if component not in self.component_status:
            return True
        
        last_check = datetime.fromisoformat(self.component_status[component]["last_check"])
        requirements = self.component_requirements[component]
        retry_interval = requirements["retry_interval"]
        
        time_since_check = datetime.now() - last_check
        return time_since_check.total_seconds() > retry_interval
    
    def get_connectivity_summary(self) -> Dict[str, Any]:
        """Get overall connectivity summary"""
        all_status = self.get_all_component_status()
        
        online_components = sum(1 for status in all_status.values() if status["status"] == "online")
        total_components = len(all_status)
        
        return {
            "overall_status": "online" if online_components > 0 else "offline",
            "online_components": online_components,
            "total_components": total_components,
            "component_details": all_status,
            "checker_info": self.checker.get_connectivity_info()
        }

# Global instances
_internet_checker = None
_connectivity_manager = None

def get_internet_checker() -> InternetChecker:
    """Get global internet checker instance"""
    global _internet_checker
    if _internet_checker is None:
        _internet_checker = InternetChecker()
    return _internet_checker

def get_connectivity_manager() -> ConnectivityManager:
    """Get global connectivity manager instance"""
    global _connectivity_manager
    if _connectivity_manager is None:
        _connectivity_manager = ConnectivityManager()
    return _connectivity_manager

async def main():
    """Test the internet checker"""
    logging.basicConfig(level=logging.INFO)
    
    checker = get_internet_checker()
    manager = get_connectivity_manager()
    
    # Test basic connectivity
    print("Testing internet connectivity...")
    status = await checker.check_connectivity_async()
    print(f"Internet status: {'Online' if status else 'Offline'}")
    
    # Test component connectivity
    print("\nTesting component connectivity...")
    components = ["web_scraping", "model_download", "api_calls", "real_time_data"]
    
    for component in components:
        result = await manager.check_component_connectivity_async(component)
        print(f"{component}: {result['status']} - {result['message']}")
    
    # Get summary
    print("\nConnectivity summary:")
    summary = manager.get_connectivity_summary()
    print(f"Overall status: {summary['overall_status']}")
    print(f"Online components: {summary['online_components']}/{summary['total_components']}")

if __name__ == "__main__":
    asyncio.run(main())
