"""
Llama 3 8B Setup Script for Routesit AI
Automated download, quantization, and validation
"""

import os
import sys
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from huggingface_hub import hf_hub_download, snapshot_download
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama3Setup:
    """Setup Llama 3 8B for Routesit AI"""
    
    def __init__(self):
        self.model_name = "meta-llama/Llama-3-8B-Instruct"
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Check system capabilities
        self.has_gpu = torch.cuda.is_available()
        self.gpu_memory = self._get_gpu_memory()
        self.cpu_cores = os.cpu_count()
        
        logger.info(f"System capabilities:")
        logger.info(f"  GPU available: {self.has_gpu}")
        logger.info(f"  GPU memory: {self.gpu_memory}GB" if self.has_gpu else "  CPU cores: {self.cpu_cores}")
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0
    
    def check_requirements(self) -> bool:
        """Check if system meets requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
        
        # Check PyTorch
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
        except ImportError:
            logger.error("PyTorch not installed")
            return False
        
        # Check transformers
        try:
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
        except ImportError:
            logger.error("Transformers not installed")
            return False
        
        # Check bitsandbytes for quantization
        try:
            import bitsandbytes
            logger.info(f"BitsAndBytes version: {bitsandbytes.__version__}")
        except ImportError:
            logger.warning("BitsAndBytes not installed - will use CPU-only mode")
        
        # Check available memory
        import psutil
        available_memory = psutil.virtual_memory().available / 1024**3
        logger.info(f"Available RAM: {available_memory:.2f}GB")
        
        if available_memory < 8:
            logger.warning("Less than 8GB RAM available - performance may be limited")
        
        return True
    
    def download_model(self) -> bool:
        """Download Llama 3 8B model"""
        logger.info(f"Downloading {self.model_name}...")
        
        try:
            # Check if model already exists
            model_path = self.models_dir / "llama-3-8b-instruct"
            if model_path.exists():
                logger.info("Model already downloaded")
                return True
            
            # Download model files
            logger.info("Downloading model files (this may take a while)...")
            
            # Download tokenizer
            tokenizer_path = hf_hub_download(
                repo_id=self.model_name,
                filename="tokenizer.json",
                cache_dir=str(self.models_dir)
            )
            
            # Download config
            config_path = hf_hub_download(
                repo_id=self.model_name,
                filename="config.json",
                cache_dir=str(self.models_dir)
            )
            
            # Download model weights (sharded)
            model_files = [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json"
            ]
            
            for filename in tqdm(model_files, desc="Downloading model weights"):
                try:
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=filename,
                        cache_dir=str(self.models_dir)
                    )
                except Exception as e:
                    logger.warning(f"Could not download {filename}: {e}")
            
            logger.info("Model download completed")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
    
    def create_quantized_model(self) -> bool:
        """Create 4-bit quantized version of the model"""
        logger.info("Creating 4-bit quantized model...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            # Model path
            model_path = self.models_dir / "llama-3-8b-instruct"
            
            if not model_path.exists():
                logger.error("Model not found. Please download first.")
                return False
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load and quantize model
            logger.info("Loading and quantizing model...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Save quantized model
            quantized_path = self.models_dir / "llama-3-8b-instruct-quantized"
            quantized_path.mkdir(exist_ok=True)
            
            model.save_pretrained(str(quantized_path))
            tokenizer.save_pretrained(str(quantized_path))
            
            logger.info(f"Quantized model saved to {quantized_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating quantized model: {e}")
            return False
    
    def validate_model(self) -> bool:
        """Validate model functionality"""
        logger.info("Validating model...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load quantized model
            model_path = self.models_dir / "llama-3-8b-instruct-quantized"
            
            if not model_path.exists():
                logger.error("Quantized model not found")
                return False
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Test inference
            test_prompt = "What is road safety?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Test response: {response[:100]}...")
            
            logger.info("Model validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def benchmark_performance(self) -> Dict[str, float]:
        """Benchmark model performance"""
        logger.info("Benchmarking model performance...")
        
        try:
            import time
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_path = self.models_dir / "llama-3-8b-instruct-quantized"
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Benchmark inference speed
            test_prompts = [
                "Analyze this road safety scenario: faded zebra crossing",
                "What interventions are needed for school zone safety?",
                "Calculate cost-benefit for speed hump installation"
            ]
            
            times = []
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True
                    )
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
            else:
                import psutil
                process = psutil.Process()
                memory_used = process.memory_info().rss / 1024**3
            
            benchmark_results = {
                "average_inference_time": avg_time,
                "memory_usage_gb": memory_used,
                "tokens_per_second": 100 / avg_time if avg_time > 0 else 0
            }
            
            logger.info(f"Benchmark results: {benchmark_results}")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}
    
    def create_model_config(self) -> bool:
        """Create model configuration file"""
        logger.info("Creating model configuration...")
        
        try:
            config = {
                "model_name": "Llama-3-8B-Instruct",
                "model_path": str(self.models_dir / "llama-3-8b-instruct-quantized"),
                "quantization": "4-bit",
                "device": "auto",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "system_capabilities": {
                    "has_gpu": self.has_gpu,
                    "gpu_memory_gb": self.gpu_memory,
                    "cpu_cores": self.cpu_cores
                },
                "performance": self.benchmark_performance()
            }
            
            config_path = Path("config/model_config.yaml")
            config_path.parent.mkdir(exist_ok=True)
            
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Model configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model config: {e}")
            return False
    
    def setup_complete(self) -> bool:
        """Complete setup process"""
        logger.info("Starting Llama 3 8B setup...")
        
        # Check requirements
        if not self.check_requirements():
            logger.error("Requirements check failed")
            return False
        
        # Download model
        if not self.download_model():
            logger.error("Model download failed")
            return False
        
        # Create quantized model
        if not self.create_quantized_model():
            logger.error("Model quantization failed")
            return False
        
        # Validate model
        if not self.validate_model():
            logger.error("Model validation failed")
            return False
        
        # Create configuration
        if not self.create_model_config():
            logger.error("Configuration creation failed")
            return False
        
        logger.info("Llama 3 8B setup completed successfully!")
        return True

def main():
    """Main setup function"""
    setup = Llama3Setup()
    
    if setup.setup_complete():
        print("\nLlama 3 8B setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python -c 'from src.core.llama3_engine import get_llm_engine; engine = get_llm_engine()'")
        print("2. Test the engine with a sample query")
        print("3. Proceed with multilingual system setup")
    else:
        print("\nSetup failed. Please check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
