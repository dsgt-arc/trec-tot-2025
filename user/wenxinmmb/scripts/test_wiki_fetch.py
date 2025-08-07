"""
Unit tests for fetch_document_from_wiki_full function.
Tests various Wikipedia entities to ensure proper content retrieval.
"""

import unittest
from unittest.mock import patch, Mock
import requests
import sys
import os
import re

# Add the current directory to Python path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module using importlib since it starts with a number
import importlib.util
spec = importlib.util.spec_from_file_location("llm_query_gen", "6_llm_query_gen.py")
llm_query_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_query_gen)

# Import the function we want to test
fetch_document_from_wiki_full = llm_query_gen.fetch_document_from_wiki_full


def count_words(text):
    """Count the number of words in text."""
    return len(text.split())


def estimate_tokens(text):
    """Estimate token count using a simple heuristic (words * 1.3)."""
    words = count_words(text)
    # Rough approximation: tokens ≈ words * 1.3 for English text
    return int(words * 1.3)


class TestFetchDocumentFromWikiFull(unittest.TestCase):
    """Test cases for the fetch_document_from_wiki_full function."""

    def test_altay_given_name(self):
        """Test fetching content for 'Altay (given name)'."""
        result = fetch_document_from_wiki_full("Altay (given name)")
        word_count = count_words(result)
        token_count = estimate_tokens(result)
        
        print(f"Altay (given name) - First 50 chars: '{result[:50]}'")
        print(f"Altay (given name) - Length: {len(result)} characters, {word_count} words, ~{token_count} tokens")

        # Stub article doesn't have content

        # Should return a string (not an error message)
        self.assertIsInstance(result, str)
        self.assertFalse(result.startswith("Failed"))
        self.assertFalse(result.startswith("No pages"))
        
        # Should contain relevant content about the name
        self.assertGreater(len(result), 50)  # Should have substantial content
        # Check for typical content that might appear in a name article
        result_lower = result.lower()
        self.assertTrue(any(word in result_lower for word in ['name', 'altay', 'given']))

    def test_miss_meyers(self):
        """Test fetching content for 'Miss Meyers'."""
        result = fetch_document_from_wiki_full("Miss Meyers")
        word_count = count_words(result)
        token_count = estimate_tokens(result)
        
        print(f"Miss Meyers - First 50 chars: '{result[:50]}'")
        print(f"Miss Meyers - Length: {len(result)} characters, {word_count} words, ~{token_count} tokens")
        
        # This might be a less common page, so we test both success and failure cases
        self.assertIsInstance(result, str)
        
        if not result.startswith("Failed") and not result.startswith("No pages"):
            # If successful, should have content
            self.assertGreater(len(result), 10)
        else:
            # If it fails, should be a proper error message
            self.assertTrue(result.startswith("Failed") or result.startswith("No pages"))

    def test_apollo_11(self):
        """Test fetching content for 'Apollo 11'."""
        result = fetch_document_from_wiki_full("Apollo 11")
        word_count = count_words(result)
        token_count = estimate_tokens(result)
        
        print(f"Apollo 11 - First 50 chars: '{result[:50]}'")
        print(f"Apollo 11 - Length: {len(result)} characters, {word_count} words, ~{token_count} tokens")
        
        # Apollo 11 is a well-known page, should definitely exist
        self.assertIsInstance(result, str)
        self.assertFalse(result.startswith("Failed"))
        self.assertFalse(result.startswith("No pages"))
        
        # Should contain substantial content about Apollo 11
        self.assertGreater(len(result), 500)  # Apollo 11 page should be substantial
        result_lower = result.lower()
        
        # Check for Apollo 11 related content
        apollo_keywords = ['apollo', 'moon', 'nasa', 'lunar', 'armstrong', 'mission']
        self.assertTrue(any(word in result_lower for word in apollo_keywords))

    def test_taylor_swift_typo(self):
        """Test fetching content for 'Taylor Switft' (intentional typo)."""
        result = fetch_document_from_wiki_full("Taylor Switft")
        word_count = count_words(result)
        token_count = estimate_tokens(result)
        
        print(f"Taylor Switft (typo) - First 50 chars: '{result[:50]}'")
        print(f"Taylor Switft (typo) - Length: {len(result)} characters, {word_count} words, ~{token_count} tokens")
        
        # This should either redirect to Taylor Swift or fail
        self.assertIsInstance(result, str)
        
        if not result.startswith("Failed") and not result.startswith("No pages"):
            # If Wikipedia redirects the typo, we might get Taylor Swift content
            self.assertGreater(len(result), 100)
        else:
            # If it fails due to typo, should be a proper error message
            self.assertTrue(result.startswith("Failed") or result.startswith("No pages"))

    def test_taylor_swift_correct(self):
        """Test fetching content for 'Taylor Swift' (correct spelling)."""
        result = fetch_document_from_wiki_full("List of Taylor Swift live performances")
        word_count = count_words(result)
        token_count = estimate_tokens(result)
        
        print(f"Taylor Swift - First 50 chars: '{result[:50]}'")
        print(f"Taylor Swift - Length: {len(result)} characters, {word_count} words, ~{token_count} tokens")
        
        # Taylor Swift is a well-known page, should definitely exist
        self.assertIsInstance(result, str)
        self.assertFalse(result.startswith("Failed"))
        self.assertFalse(result.startswith("No pages"))
        
        # Should contain substantial content about Taylor Swift
        self.assertGreater(len(result), 500)  # Should be a substantial article
        result_lower = result.lower()
        
        # Check for Taylor Swift related content
        swift_keywords = ['taylor', 'swift', 'singer', 'music', 'album', 'song']
        self.assertTrue(any(word in result_lower for word in swift_keywords))

    def test_mercury_planet(self):
        """Test fetching content for 'Mercury (planet)'."""
        result = fetch_document_from_wiki_full("Mercury (planet)")
        word_count = count_words(result)
        token_count = estimate_tokens(result)
        
        print(f"Mercury (planet) - First 50 chars: '{result[:50]}'")
        print(f"Mercury (planet) - Length: {len(result)} characters, {word_count} words, ~{token_count} tokens")
        
        # Mercury is a well-known planet, should definitely exist
        self.assertIsInstance(result, str)
        self.assertFalse(result.startswith("Failed"))
        self.assertFalse(result.startswith("No pages"))
        
        # Should contain substantial content about Mercury
        self.assertGreater(len(result), 500)  # Mercury page should be substantial
        result_lower = result.lower()
        
        # Check for Mercury planet related content
        mercury_keywords = ['mercury', 'planet', 'sun', 'solar system', 'orbit', 'innermost']
        self.assertTrue(any(word in result_lower for word in mercury_keywords))

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
