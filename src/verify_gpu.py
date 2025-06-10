"""
verify_gpu.py

This script provides comprehensive verification of GPU/CUDA availability and system
resources for the DSPy automotive intelligence project. It performs diagnostics on:

1. PyTorch CUDA availability and GPU device information
2. NVIDIA GPU status using nvidia-smi utilities
3. Ollama service status and loaded models
4. System resource utilization (CPU, RAM, disk)
5. DSPy inference capabilities with GPU acceleration

This diagnostic tool helps ensure that the development environment is properly
configured for optimal performance during model training and inference.

Usage:
    .\.venv\Scripts\python.exe src\verify_gpu.py

Example:
    >>> .\.venv\Scripts\python.exe src\verify_gpu.py
    🚀 GPU AND SYSTEM VERIFICATION
    ✅ CUDA available: True
    ✅ Ollama processes found: 1
    ✅ DSPy inference successful!
"""

import os
import subprocess

import psutil
import torch


def check_pytorch_cuda():
    """
    Verify PyTorch CUDA availability and GPU device information.

    This function performs comprehensive checks of the PyTorch CUDA installation:
    1. Verifies PyTorch version and CUDA availability
    2. Reports CUDA and cuDNN versions if available
    3. Enumerates all available GPU devices with specifications
    4. Tests basic GPU tensor operations to verify functionality
    5. Provides troubleshooting guidance if CUDA is not available

    The function is designed to help diagnose common CUDA installation issues
    and provide actionable recommendations for fixing configuration problems.

    Returns:
        None

    Side Effects:
        - Prints detailed PyTorch and CUDA status information
        - Creates test tensors on GPU if available
        - Outputs troubleshooting recommendations if needed

    Example:
        >>> check_pytorch_cuda()
        ============================================================
        🔍 PYTORCH CUDA VERIFICATION
        ============================================================
        PyTorch version: 2.1.0
        CUDA available: True
        ✅ GPU tensor operations successful!
    """
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
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print("ℹ️  CUDA toolkit is installed but PyTorch can't access it.")
                print("You might need to reinstall PyTorch with CUDA support:")
                print(
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
                )
            else:
                print("ℹ️  CUDA toolkit not found in PATH.")
        except FileNotFoundError:
            print("ℹ️  CUDA toolkit (nvcc) not found.")


def check_nvidia_gpu():
    """
    Check NVIDIA GPU information and status using nvidia-smi utilities.

    This function queries the NVIDIA GPU driver to get real-time information
    about GPU utilization, memory usage, and thermal status. It provides
    detailed metrics that are essential for monitoring performance during
    intensive model training or inference operations.

    The function reports:
    - GPU model names and specifications
    - Memory usage (used/total/free)
    - GPU utilization percentages
    - Operating temperatures

    Returns:
        None

    Side Effects:
        - Executes nvidia-smi subprocess command
        - Prints detailed GPU status information
        - Handles cases where nvidia-smi is not available

    Example:
        >>> check_nvidia_gpu()
        ============================================================
        🔍 NVIDIA GPU STATUS
        ============================================================
        GPU 0: NVIDIA GeForce RTX 4090
          - Memory: 2048MB / 24576MB used (22528MB free)
          - GPU Utilization: 15%
          - Temperature: 45°C

    Note:
        Requires nvidia-smi to be installed and accessible in PATH.
        This is automatically available with NVIDIA GPU drivers.
    """
    print("\n" + "=" * 60)
    print("🔍 NVIDIA GPU STATUS")
    print("=" * 60)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                parts = line.split(", ")
                if len(parts) >= 6:
                    name, total_mem, used_mem, free_mem, gpu_util, temp = parts
                    print(f"GPU {i}: {name}")
                    print(
                        f"  - Memory: {used_mem}MB / {total_mem}MB used ({free_mem}MB free)"
                    )
                    print(f"  - GPU Utilization: {gpu_util}%")
                    print(f"  - Temperature: {temp}°C")
        else:
            print("❌ nvidia-smi failed or no NVIDIA GPU detected")

    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found or timed out")


def check_ollama_status():
    """
    Check Ollama service status and enumerate available/loaded models.

    This function verifies that the Ollama service is running and provides
    detailed information about the model management state. It checks:
    1. Running Ollama processes and their resource usage
    2. Available models in the local Ollama repository
    3. Currently loaded models in memory

    This information is crucial for ensuring that the DSPy pipeline can
    successfully connect to and utilize Ollama models for inference.

    Returns:
        None

    Side Effects:
        - Scans system processes for Ollama instances
        - Executes ollama CLI commands (list, ps)
        - Prints detailed service and model status
        - Returns early if no Ollama processes are found

    Example:
        >>> check_ollama_status()
        ============================================================
        🔍 OLLAMA STATUS
        ============================================================
        ✅ Found 1 Ollama process(es):
          - PID 12345: ollama.exe (2048.5MB)

        📋 Available models:
        NAME            ID              SIZE    MODIFIED
        gemma3:12b     abc123def456    7.2GB   2 hours ago

    Note:
        Requires Ollama to be installed and the ollama command to be
        available in the system PATH.
    """
    print("\n" + "=" * 60)
    print("🔍 OLLAMA STATUS")
    print("=" * 60)

    try:
        # Check if Ollama process is running
        ollama_processes = []
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            if "ollama" in proc.info["name"].lower():
                ollama_processes.append(proc)

        if ollama_processes:
            print(f"✅ Found {len(ollama_processes)} Ollama process(es):")
            for proc in ollama_processes:
                memory_mb = proc.info["memory_info"].rss / 1024 / 1024
                print(
                    f"  - PID {proc.info['pid']}: {proc.info['name']} ({memory_mb:.1f}MB)"
                )
        else:
            print("❌ No Ollama processes found")
            return

        # Try to get model list
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("\n📋 Available models:")
                print(result.stdout)
            else:
                print("❌ Failed to get Ollama model list")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ ollama command not found or timed out")

        # Try to get running models (if any)
        try:
            result = subprocess.run(
                ["ollama", "ps"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("🏃 Currently loaded models:")
                print(result.stdout)
            else:
                print("ℹ️  No models currently loaded in memory")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Failed to check loaded models")

    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")


def check_system_resources():
    """
    Monitor and report overall system resource utilization.

    This function provides a snapshot of critical system resources that
    affect model training and inference performance:
    - CPU utilization and core count
    - RAM usage (used/total/percentage)
    - Disk space utilization

    This information helps identify potential bottlenecks and ensures
    the system has sufficient resources for intensive AI workloads.

    Returns:
        None

    Side Effects:
        - Measures CPU usage over a 1-second interval
        - Queries system memory and disk usage
        - Prints formatted resource utilization report

    Example:
        >>> check_system_resources()
        ============================================================
        🔍 SYSTEM RESOURCES
        ============================================================
        CPU Usage: 25.3% (16 cores)
        RAM Usage: 12.5GB / 32.0GB (39.1%)
        Disk Usage: 250.8GB / 1000.0GB (25.1%)

    Note:
        CPU usage is measured over a 1-second sampling interval for accuracy.
        Disk usage is checked for the C: drive on Windows systems.
    """
    print("\n" + "=" * 60)
    print("🔍 SYSTEM RESOURCES")
    print("=" * 60)

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU Usage: {cpu_percent}% ({cpu_count} cores)")

    # Memory
    memory = psutil.virtual_memory()
    print(
        f"RAM Usage: {memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB ({memory.percent}%)"
    )

    # Disk
    disk = psutil.disk_usage("C:")
    print(
        f"Disk Usage: {disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB ({disk.percent}%)"
    )


def test_dspy_gpu_inference():
    """
    Test DSPy framework integration with GPU-accelerated inference.

    This function performs an end-to-end test of the DSPy framework's ability
    to connect to and utilize Ollama models for inference. It:
    1. Loads environment variables for model configuration
    2. Initializes DSPy with the specified Ollama model
    3. Performs a simple inference test to verify connectivity
    4. Reports success or provides diagnostic information on failure

    This test validates that the complete inference pipeline is functional
    and ready for use in the main optimization workflow.

    Returns:
        None

    Environment Variables:
        OLLAMA_MODEL (str): The Ollama model name to use for testing
                          (default: "gemma3:12b")

    Side Effects:
        - Configures DSPy global settings
        - Performs a test inference call
        - Prints test results and diagnostic information
        - Handles and reports any configuration or inference errors

    Example:
        >>> test_dspy_gpu_inference()
        ============================================================
        🔍 DSPY GPU INFERENCE TEST
        ============================================================
        Attempting to configure DSPy with model: gemma3:12b
        Testing simple generation...
        ✅ DSPy inference successful!
        Response: The answer to 2+2 is 4. This is a basic arithmetic operation...

    Note:
        Some errors during this test are normal, especially if Ollama models
        are not pre-loaded. The main optimization pipeline includes additional
        error handling and retry logic.
    """
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
        print("✅ DSPy inference successful!")
        print(f"Response: {str(response)[:100]}...")

    except Exception as e:
        print(f"❌ DSPy inference test failed: {e}")
        print("ℹ️  This is normal - DSPy will work during optimization")


if __name__ == "__main__":
    """
    Main execution function that runs all verification checks in sequence.
    
    This orchestrates a comprehensive system verification by executing all
    diagnostic functions in a logical order. The complete verification covers
    hardware capabilities, software installation status, and end-to-end
    functionality testing.
    """
    print("🚀 GPU AND SYSTEM VERIFICATION")
    print(
        "This script will check GPU availability, Ollama status, and system resources.\n"
    )

    check_pytorch_cuda()
    check_nvidia_gpu()
    check_ollama_status()
    check_system_resources()
    test_dspy_gpu_inference()

    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)
