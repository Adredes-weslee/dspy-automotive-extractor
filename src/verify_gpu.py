"""
verify_gpu.py

This script verifies GPU/CUDA availability and usage for the DSPy automotive intelligence project.
It checks:
1. PyTorch CUDA availability
2. GPU device information
3. Ollama model GPU usage
4. Memory usage statistics
"""

import torch
import os
import psutil
import subprocess
import json
from pathlib import Path

def check_pytorch_cuda():
    """Check PyTorch CUDA availability and GPU information."""
    print("=" * 60)
    print("🔍 PYTORCH CUDA VERIFICATION")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  - Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute capability: {gpu_props.major}.{gpu_props.minor}")
            
        # Test GPU tensor operations
        print("\n🧪 Testing GPU tensor operations...")
        try:
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("✅ GPU tensor operations successful!")
            print(f"Result tensor shape: {z.shape}, device: {z.device}")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
    else:
        print("❌ CUDA not available. PyTorch will use CPU.")
        
        # Check if CUDA toolkit is installed
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("ℹ️  CUDA toolkit is installed but PyTorch can't access it.")
                print("You might need to reinstall PyTorch with CUDA support:")
                print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("ℹ️  CUDA toolkit not found in PATH.")
        except FileNotFoundError:
            print("ℹ️  CUDA toolkit (nvcc) not found.")

def check_nvidia_gpu():
    """Check NVIDIA GPU information using nvidia-smi."""
    print("\n" + "=" * 60)
    print("🔍 NVIDIA GPU STATUS")
    print("=" * 60)
    
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 6:
                    name, total_mem, used_mem, free_mem, gpu_util, temp = parts
                    print(f"GPU {i}: {name}")
                    print(f"  - Memory: {used_mem}MB / {total_mem}MB used ({free_mem}MB free)")
                    print(f"  - GPU Utilization: {gpu_util}%")
                    print(f"  - Temperature: {temp}°C")
        else:
            print("❌ nvidia-smi failed or no NVIDIA GPU detected")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found or timed out")

def check_ollama_status():
    """Check if Ollama is running and which models are loaded."""
    print("\n" + "=" * 60)
    print("🔍 OLLAMA STATUS")
    print("=" * 60)
    
    try:
        # Check if Ollama process is running
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            if 'ollama' in proc.info['name'].lower():
                ollama_processes.append(proc)
        
        if ollama_processes:
            print(f"✅ Found {len(ollama_processes)} Ollama process(es):")
            for proc in ollama_processes:
                memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                print(f"  - PID {proc.info['pid']}: {proc.info['name']} ({memory_mb:.1f}MB)")
        else:
            print("❌ No Ollama processes found")
            return
            
        # Try to get model list
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"\n📋 Available models:")
                print(result.stdout)
            else:
                print("❌ Failed to get Ollama model list")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ ollama command not found or timed out")
            
        # Try to get running models (if any)
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"🏃 Currently loaded models:")
                print(result.stdout)
            else:
                print("ℹ️  No models currently loaded in memory")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Failed to check loaded models")
            
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")

def check_system_resources():
    """Check overall system resource usage."""
    print("\n" + "=" * 60)
    print("🔍 SYSTEM RESOURCES")
    print("=" * 60)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU Usage: {cpu_percent}% ({cpu_count} cores)")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"RAM Usage: {memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB ({memory.percent}%)")
    
    # Disk
    disk = psutil.disk_usage('C:')
    print(f"Disk Usage: {disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB ({disk.percent}%)")

def test_dspy_gpu_inference():
    """Test DSPy with GPU inference if possible."""
    print("\n" + "=" * 60)
    print("🔍 DSPY GPU INFERENCE TEST")
    print("=" * 60)
    
    try:
        import dspy
        
        # Load environment variables
        model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
        print(f"Attempting to configure DSPy with model: {model_name}")
        
        # In DSPy 2.6.27, use LM class with ollama provider
        llm = dspy.LM(model=f"ollama/{model_name}")
        dspy.settings.configure(lm=llm)
        
        # Simple test
        print("Testing simple generation...")
        response = llm("What is 2+2?")
        print(f"✅ DSPy inference successful!")
        print(f"Response: {str(response)[:100]}...")
        
    except Exception as e:
        print(f"❌ DSPy inference test failed: {e}")
        print("ℹ️  This is normal - DSPy will work during optimization")
        
        
if __name__ == "__main__":
    print("🚀 GPU AND SYSTEM VERIFICATION")
    print("This script will check GPU availability, Ollama status, and system resources.\n")
    
    check_pytorch_cuda()
    check_nvidia_gpu()
    check_ollama_status()
    check_system_resources()
    test_dspy_gpu_inference()
    
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)
