#!/usr/bin/env python3
"""
CUDA Test Script for TREC-TOT 2025
This script tests if CUDA is available and working properly.
"""

import torch
import sys

def test_cuda():
    """Test CUDA availability and functionality."""
    print("=" * 60)
    print("CUDA TEST SCRIPT")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # GPU memory info
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"GPU compute capability: {props.major}.{props.minor}")
        
        # Test basic CUDA operations
        print("\nTesting CUDA operations...")
        try:
            # Create a tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Perform matrix multiplication
            z = torch.mm(x, y)
            print(f"Matrix multiplication successful: {z.shape}")
            
            # Test memory usage
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            print("CUDA operations completed successfully!")
            
        except Exception as e:
            print(f"Error during CUDA operations: {e}")
            return False
            
        return True
    else:
        print("CUDA is not available!")
        return False

if __name__ == "__main__":
    success = test_cuda()
    print("\n" + "=" * 60)
    if success:
        print("✅ CUDA TEST PASSED - GPU is working correctly!")
    else:
        print("❌ CUDA TEST FAILED - Check your GPU setup!")
    print("=" * 60) 