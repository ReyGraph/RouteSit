#!/usr/bin/env python3
"""
Llama 3 8B Local Setup Script
Downloads and configures Llama 3 8B with 4-bit quantization for local deployment
Requires HuggingFace authentication token
"""

import os
import sys
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from huggingface_hub import hf_hub_download, snapshot_download, login
import requests
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)

class Llama3LocalSetup:
    """Setup Llama 3 8B for local deployment with quantization"""
    
    def __init__(self):
        self.model_name = "meta-llama/Llama-3-8B-Instruct"
        self.model_dir = Path("models/llama3_8b")
        self.cache_dir = Path("models/cache")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Quantization config
        self.quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    
    def check_huggingface_auth(self) -> bool:
        """Check if HuggingFace authentication is available"""
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            logger.info(f"Authenticated as: {user_info['name']}")
            return True
        except Exception as e:
            logger.error(f"HuggingFace authentication failed: {e}")
            return False
    
    def authenticate_huggingface(self, token: Optional[str] = None) -> bool:
        """Authenticate with HuggingFace"""
        try:
            if token:
                login(token=token)
            else:
                # Try to login interactively
                login()
            
            return self.check_huggingface_auth()
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def download_model(self) -> bool:
        """Download Llama 3 8B model"""
        try:
            logger.info(f"Downloading {self.model_name}...")
            
            # Download model files
            model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=self.cache_dir,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Model downloaded to: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def test_model_loading(self) -> bool:
        """Test if the model can be loaded with quantization"""
        try:
            logger.info("Testing model loading with quantization...")
            
            from transformers import (
                AutoTokenizer, 
                AutoModelForCausalLM, 
                BitsAndBytesConfig
            )
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.quantization_config["load_in_4bit"],
                bnb_4bit_compute_dtype=getattr(torch, self.quantization_config["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=self.quantization_config["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=self.quantization_config["bnb_4bit_quant_type"]
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True
            )
            
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Test inference
            test_prompt = "What is road safety?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Test response: {response}")
            
            # Save model info
            model_info = {
                "model_name": self.model_name,
                "model_path": str(self.model_dir),
                "quantization": self.quantization_config,
                "device": self.device,
                "test_successful": True,
                "memory_usage": self._get_memory_usage()
            }
            
            with open(self.model_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info("Model loading test successful!")
            return True
            
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        try:
            if torch.cuda.is_available():
                return {
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                }
            else:
                import psutil
                process = psutil.Process()
                return {
                    "cpu_memory_used": process.memory_info().rss / 1024**3,  # GB
                    "cpu_memory_percent": process.memory_percent()
                }
        except Exception:
            return {"error": "Could not determine memory usage"}
    
    def create_fallback_model(self) -> bool:
        """Create fallback model configuration"""
        try:
            logger.info("Creating fallback model configuration...")
            
            fallback_config = {
                "model_name": "microsoft/DialoGPT-medium",
                "fallback_reason": "Llama 3 8B not available",
                "quantization": False,
                "device": "cpu"
            }
            
            with open(self.model_dir / "fallback_config.json", 'w') as f:
                json.dump(fallback_config, f, indent=2)
            
            logger.info("Fallback configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback config: {e}")
            return False
    
    def setup_model_config(self) -> bool:
        """Setup model configuration for Routesit AI"""
        try:
            logger.info("Setting up model configuration...")
            
            config = {
                "model_name": self.model_name,
                "model_path": str(self.model_dir),
                "quantization": self.quantization_config,
                "device": self.device,
                "max_length": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "road_safety_prompts": {
                    "system_prompt": "You are Routesit AI, an expert road safety analyst specializing in intervention recommendations, cascading effects analysis, and implementation planning for Indian roads.",
                    "intervention_analysis": "Analyze this road safety situation and recommend appropriate interventions:",
                    "cascading_effects": "Predict the cascading effects of this intervention:",
                    "cost_benefit": "Perform cost-benefit analysis for this intervention:",
                    "implementation": "Create an implementation plan for this intervention:"
                }
            }
            
            config_path = Path("config/llama3_config.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Model configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup model config: {e}")
            return False
    
    def run_setup(self, hf_token: Optional[str] = None) -> bool:
        """Run complete setup process"""
        logger.info("Starting Llama 3 8B local setup...")
        
        # Step 1: Check authentication
        if not self.check_huggingface_auth():
            logger.info("HuggingFace authentication required...")
            if not self.authenticate_huggingface(hf_token):
                logger.error("Authentication failed. Please provide a valid HuggingFace token.")
                logger.info("Creating fallback configuration...")
                self.create_fallback_model()
                return False
        
        # Step 2: Download model
        if not self.download_model():
            logger.error("Model download failed")
            self.create_fallback_model()
            return False
        
        # Step 3: Test model loading
        if not self.test_model_loading():
            logger.error("Model loading test failed")
            self.create_fallback_model()
            return False
        
        # Step 4: Setup configuration
        if not self.setup_model_config():
            logger.error("Configuration setup failed")
            return False
        
        logger.info("Llama 3 8B setup completed successfully!")
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Setup Llama 3 8B for local deployment")
    parser.add_argument("--token", help="HuggingFace authentication token")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run setup
    setup = Llama3LocalSetup()
    success = setup.run_setup(args.token)
    
    if success:
        print("\nLlama 3 8B setup completed successfully!")
        print("Model is ready for fine-tuning and deployment.")
    else:
        print("\nSetup failed. Check logs for details.")
        print("Fallback configuration has been created.")
        sys.exit(1)

if __name__ == "__main__":
    main()
