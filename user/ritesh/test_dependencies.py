#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed and CUDA is available
"""

import sys
import time

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    
    try:
        import json
        import numpy as np
        import pandas as pd
        import time
        import os
        from datetime import datetime
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_sentence_transformers():
    """Test sentence-transformers library"""
    print("\nTesting sentence-transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers imported successfully")
        
        # Test model loading
        print("Testing model loading...")
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        print(f"✓ Model loaded successfully")
        print(f"  Model dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test encoding
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_texts, show_progress_bar=False)
        print(f"✓ Encoding successful, shape: {embeddings.shape}")
        
        return True
    except ImportError as e:
        print(f"✗ sentence-transformers import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ sentence-transformers test failed: {e}")
        return False

def test_faiss():
    """Test FAISS library"""
    print("\nTesting FAISS...")
    
    try:
        import faiss
        import numpy as np
        print("✓ FAISS imported successfully")
        
        # Test basic FAISS operations
        dimension = 768
        index = faiss.IndexFlatIP(dimension)
        print(f"✓ FAISS index created successfully (dimension: {dimension})")
        
        # Test adding vectors
        test_vectors = np.random.random((10, dimension)).astype('float32')
        index.add(test_vectors)
        print(f"✓ Added {len(test_vectors)} vectors to index")
        
        # Test search
        query_vector = np.random.random((1, dimension)).astype('float32')
        scores, indices = index.search(query_vector, 3)
        print(f"✓ Search successful, found {len(indices[0])} results")
        
        return True
    except ImportError as e:
        print(f"✗ FAISS import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ FAISS test failed: {e}")
        return False

def test_tqdm():
    """Test tqdm library"""
    print("\nTesting tqdm...")
    
    try:
        from tqdm import tqdm
        print("✓ tqdm imported successfully")
        
        # Test progress bar
        for i in tqdm(range(5), desc="Testing progress bar"):
            time.sleep(0.1)
        print("✓ Progress bar working")
        
        return True
    except ImportError as e:
        print(f"✗ tqdm import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ tqdm test failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test GPU tensor operations
            device = torch.device('cuda')
            test_tensor = torch.randn(1000, 768).to(device)
            print(f"✓ GPU tensor operations successful")
            
            return True
        else:
            print("⚠ CUDA is not available (CPU only)")
            print("  This will work but will be slower for large datasets")
            return True  # Not a failure, just slower
            
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_sentence_transformers_cuda():
    """Test sentence-transformers with CUDA"""
    print("\nTesting sentence-transformers with CUDA...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        if torch.cuda.is_available():
            # Test model on GPU
            model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')
            print("✓ Model loaded on GPU successfully")
            
            # Test encoding on GPU
            test_texts = ["GPU test sentence 1.", "GPU test sentence 2."]
            embeddings = model.encode(test_texts, show_progress_bar=False)
            print(f"✓ GPU encoding successful, shape: {embeddings.shape}")
            
            return True
        else:
            print("⚠ Skipping GPU test (CUDA not available)")
            return True
            
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_parquet():
    """Test Parquet file operations"""
    print("\nTesting Parquet operations...")
    
    try:
        import pandas as pd
        
        # Create test data
        test_data = {
            'id': [1, 2, 3],
            'title': ['Test 1', 'Test 2', 'Test 3'],
            'text': ['Content 1', 'Content 2', 'Content 3'],
            'emb_0': [0.1, 0.2, 0.3],
            'emb_1': [0.4, 0.5, 0.6]
        }
        df = pd.DataFrame(test_data)
        
        # Test writing
        test_file = 'test_embeddings.parquet'
        df.to_parquet(test_file, index=False)
        print(f"✓ Parquet write successful: {test_file}")
        
        # Test reading
        df_read = pd.read_parquet(test_file)
        print(f"✓ Parquet read successful, shape: {df_read.shape}")
        
        # Clean up
        import os
        os.remove(test_file)
        print("✓ Test file cleaned up")
        
        return True
    except ImportError as e:
        print(f"✗ Parquet test failed (missing dependency): {e}")
        return False
    except Exception as e:
        print(f"✗ Parquet test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("DEPENDENCY AND CUDA TEST SCRIPT")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Sentence Transformers", test_sentence_transformers),
        ("FAISS", test_faiss),
        ("tqdm", test_tqdm),
        ("CUDA", test_cuda),
        ("Sentence Transformers + CUDA", test_sentence_transformers_cuda),
        ("Parquet", test_parquet),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! Your environment is ready for the embeddings project.")
    else:
        print("⚠ Some tests failed. Please install missing dependencies.")
        print("\nTo install missing dependencies, run:")
        print("pip install sentence-transformers faiss-cpu tqdm torch pandas pyarrow")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 