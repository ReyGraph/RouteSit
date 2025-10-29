#!/usr/bin/env python3
"""
Manual Mistral 7B GGUF Download Script
Downloads Mistral 7B Instruct v0.2 GGUF model for local use
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MistralDownloader:
    """Downloads Mistral 7B GGUF model manually"""
    
    def __init__(self, model_dir: str = "models/llm"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Alternative download URLs (mirrors)
        self.download_urls = [
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf",
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
        ]
        
        self.model_filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        self.model_path = self.model_dir / self.model_filename
    
    def check_existing_model(self) -> bool:
        """Check if model already exists"""
        if self.model_path.exists():
            file_size = self.model_path.stat().st_size
            logger.info(f"Model already exists: {self.model_path} ({file_size / (1024**3):.1f} GB)")
            return True
        return False
    
    def download_model(self, url: str = None) -> bool:
        """Download the model file"""
        
        if self.check_existing_model():
            return True
        
        if not url:
            url = self.download_urls[0]  # Use first URL by default
        
        logger.info(f"Downloading Mistral 7B GGUF model from: {url}")
        logger.info(f"Target file: {self.model_path}")
        
        try:
            # Create a session for better connection handling
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Start download with streaming
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size > 0:
                logger.info(f"File size: {total_size / (1024**3):.1f} GB")
            
            # Download with progress
            downloaded = 0
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # Every 10MB
                                logger.info(f"Downloaded: {downloaded / (1024**3):.1f} GB ({progress:.1f}%)")
            
            logger.info("Download completed successfully!")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            # Clean up partial file
            if self.model_path.exists():
                self.model_path.unlink()
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            if self.model_path.exists():
                self.model_path.unlink()
            return False
    
    def verify_model(self) -> bool:
        """Verify downloaded model file"""
        if not self.model_path.exists():
            logger.error("Model file not found")
            return False
        
        file_size = self.model_path.stat().st_size
        
        # Check if file size is reasonable (GGUF files are typically 4-8GB)
        if file_size < 1024**3:  # Less than 1GB
            logger.error(f"Model file too small: {file_size} bytes")
            return False
        
        if file_size > 20 * 1024**3:  # More than 20GB
            logger.error(f"Model file too large: {file_size} bytes")
            return False
        
        logger.info(f"Model verification passed: {file_size / (1024**3):.1f} GB")
        return True
    
    def create_model_info(self) -> dict:
        """Create model information file"""
        info = {
            "model_name": "Mistral-7B-Instruct-v0.2",
            "model_type": "GGUF",
            "quantization": "Q4_K_M",
            "file_path": str(self.model_path),
            "file_size_gb": self.model_path.stat().st_size / (1024**3) if self.model_path.exists() else 0,
            "download_date": str(Path().cwd()),
            "recommended_ram_gb": 8,
            "recommended_vram_gb": 6,
            "context_length": 32768,
            "parameters": "7B",
            "license": "Apache 2.0",
            "source": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
        }
        
        info_path = self.model_dir / "model_info.json"
        try:
            import json
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            logger.info(f"Model info saved to: {info_path}")
        except Exception as e:
            logger.error(f"Failed to save model info: {e}")
        
        return info
    
    def test_model_loading(self) -> bool:
        """Test if the model can be loaded"""
        try:
            from llama_cpp import Llama
            
            logger.info("Testing model loading...")
            llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,
                n_gpu_layers=0,  # CPU only for testing
                verbose=False
            )
            
            # Test generation
            response = llm("Hello, how are you?", max_tokens=50)
            logger.info(f"Test response: {response['choices'][0]['text'][:100]}...")
            
            logger.info("Model loading test passed!")
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False

def main():
    """Main download function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Mistral 7B GGUF Model Downloader")
    print("=" * 40)
    
    downloader = MistralDownloader()
    
    # Check if model already exists
    if downloader.check_existing_model():
        print("Model already exists!")
        if downloader.verify_model():
            print("Model verification passed.")
            downloader.create_model_info()
            
            # Test loading
            if downloader.test_model_loading():
                print("Model loading test passed!")
                print("Ready to use with Routesit AI!")
            else:
                print("Model loading test failed. Check llama-cpp-python installation.")
        else:
            print("Model verification failed. Re-downloading...")
            downloader.model_path.unlink()
    
    # Download model
    print("\nStarting download...")
    success = False
    
    for i, url in enumerate(downloader.download_urls):
        print(f"\nTrying download URL {i+1}/{len(downloader.download_urls)}")
        if downloader.download_model(url):
            success = True
            break
        else:
            print(f"Download failed with URL {i+1}")
    
    if success:
        print("\nDownload completed!")
        
        # Verify model
        if downloader.verify_model():
            print("Model verification passed!")
            
            # Create info file
            info = downloader.create_model_info()
            print(f"Model info: {info['model_name']} ({info['file_size_gb']:.1f} GB)")
            
            # Test loading
            print("\nTesting model loading...")
            if downloader.test_model_loading():
                print("Model loading test passed!")
                print("\n✅ Mistral 7B GGUF model is ready to use!")
                print(f"Model path: {downloader.model_path}")
                print("You can now run Routesit AI with local LLM support.")
            else:
                print("❌ Model loading test failed.")
                print("Please install llama-cpp-python: pip install llama-cpp-python")
        else:
            print("❌ Model verification failed.")
    else:
        print("❌ All download attempts failed.")
        print("Please check your internet connection and try again.")
        print("\nAlternative: Download manually from:")
        for url in downloader.download_urls:
            print(f"  {url}")

if __name__ == "__main__":
    main()
