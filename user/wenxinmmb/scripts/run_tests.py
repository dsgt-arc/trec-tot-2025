#!/usr/bin/env python3
"""
Simple test runner for the Wikipedia fetch function tests.
Run this script to execute all the unit tests.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the tests
if __name__ == '__main__':
    import test_wiki_fetch
    import unittest
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_wiki_fetch)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
