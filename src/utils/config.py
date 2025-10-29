import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration management for Routesit AI"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files"""
        config_files = {
            'model': 'model_config.yaml',
            'database': 'database_config.yaml',
            'app': 'app_config.yaml'
        }
        
        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._configs[config_name] = yaml.safe_load(f)
            else:
                print(f"Warning: Config file {config_path} not found")
                self._configs[config_name] = {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('model.llm.name')
        """
        keys = key_path.split('.')
        value = self._configs
        
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config_dict = self._configs
        
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]
        
        config_dict[keys[-1]] = value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self._configs.get('model', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self._configs.get('database', {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration"""
        return self._configs.get('app', {})
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
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

# Global config instance
config = Config()
