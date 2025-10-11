"""
Topic Classification Module for TREC-ToT 2025 Pipeline

This module handles topic classification of queries using ModernBERT-web-topics model.
Optimized for short query texts rather than long Wikipedia articles.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import pipeline
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TopicPrediction:
    """Represents a topic prediction result."""
    text: str
    predicted_topic: str
    confidence: float
    processing_time: Optional[float] = None


class TopicClassifierConfig:
    """Configuration for topic classification."""
    
    def __init__(
        self,
        model_name: str = "davanstrien/ModernBERT-web-topics-1m",
        batch_size: int = 32,
        confidence_threshold: float = 0.1,
        max_length: int = 512,
        device: Optional[Union[str, int]] = None,
        fallback_topic: str = "entertainment"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.fallback_topic = fallback_topic


class TopicClassifier:
    """
    Topic classifier for queries using ModernBERT-web-topics model.
    """
    
    # Available topics based on FAISS indexes
    AVAILABLE_TOPICS = {
        "adult_content", "art_design", "crime_law", "education_jobs",
        "electronics_hardware", "entertainment", "fashion_beauty", 
        "finance_business", "food_dining", "games", "health",
        "history_geography", "home_hobbies", "industrial", "literature",
        "politics", "religion", "science_math_technology", "social_life",
        "software", "software_development", "sports_fitness", 
        "transportation", "travel_tourism"
    }
    
    def __init__(self, config: Optional[TopicClassifierConfig] = None):
        self.config = config or TopicClassifierConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classifier = None
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the ModernBERT classifier pipeline."""
        try:
            self.logger.info(f"Initializing classifier with model: {self.config.model_name}")
            self.classifier = pipeline(
                "text-classification",
                model=self.config.model_name,
                tokenizer=self.config.model_name,
                device=self.config.device,
                truncation=True,
                max_length=self.config.max_length
            )
            self.logger.info(f"Classifier initialized successfully on device: {self.config.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize classifier: {e}")
            raise
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query text for topic classification.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query text
        """
        if not isinstance(query, str):
            return ""
        
        # Clean and normalize the query
        cleaned = query.strip()
        
        # Remove excessive whitespace and newlines
        cleaned = ' '.join(cleaned.split())
        
        # # Truncate if too long (keeping some buffer for tokenization)
        # max_chars = self.config.max_length * 4  # Rough estimate for char-to-token ratio
        # if len(cleaned) > max_chars:
        #     cleaned = cleaned[:max_chars]
        #     self.logger.debug(f"Truncated query to {max_chars} characters")
        
        return cleaned
    
    def _validate_topic(self, predicted_topic: str) -> str:
        """
        Validate and normalize predicted topic.
        
        Args:
            predicted_topic: Topic predicted by the model
            
        Returns:
            Validated topic name
        """
        # Normalize topic name
        normalized = predicted_topic.lower().split('includes')[0]
        
        # Try to find matches
        for topic in self.AVAILABLE_TOPICS:
            topic_parts = topic.split("_")
            if all(t in normalized for t in topic_parts):
                self.logger.info(f"Mapped '{normalized.split('-')[0].strip()}' to '{topic}'")
                return topic
        
        # Fallback to default topic
        self.logger.warning(f"Topic '{predicted_topic}' not found in available topics. ")
        raise
    
    def classify_single(self, query: str) -> TopicPrediction:
        """
        Classify a single query.
        
        Args:
            query: Query text to classify
            
        Returns:
            TopicPrediction object with results
        """
        start_time = time.time()
        
        try:
            # Preprocess query
            preprocessed = self._preprocess_query(query)
            
            if not preprocessed:
                return TopicPrediction(
                    text=query,
                    predicted_topic=self.config.fallback_topic,
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Get prediction
            predictions = self.classifier(preprocessed, top_k=1)
            
            if not predictions or not predictions[0]:
                raise ValueError("No predictions returned from classifier")
            
            prediction = predictions[0]
            predicted_topic = self._validate_topic(prediction["label"])
            confidence = float(prediction["score"])
            
            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                self.logger.debug(
                    f"Low confidence ({confidence:.3f}) for query, "
                    f"using fallback topic: {self.config.fallback_topic}"
                )
                predicted_topic = self.config.fallback_topic
            
            return TopicPrediction(
                text=query,
                predicted_topic=predicted_topic,
                confidence=confidence,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error classifying query: {e}")
            return TopicPrediction(
                text=query,
                predicted_topic=self.config.fallback_topic,
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def classify_batch(self, queries: List[str]) -> List[TopicPrediction]:
        """
        Classify multiple queries in batches.
        
        Args:
            queries: List of query texts to classify
            
        Returns:
            List of TopicPrediction objects
        """
        if not queries:
            return []
        
        self.logger.info(f"Classifying {len(queries)} queries in batches of {self.config.batch_size}")
        
        results = []
        total_start_time = time.time()
        
        # Process in batches
        for i in tqdm(range(0, len(queries), self.config.batch_size), 
                     desc="Topic classification"):
            batch = queries[i:i + self.config.batch_size]
            batch_start_time = time.time()
            
            try:
                # Preprocess batch
                preprocessed_batch = [self._preprocess_query(q) for q in batch]
                
                # Filter out empty queries
                valid_indices = [j for j, q in enumerate(preprocessed_batch) if q]
                valid_queries = [preprocessed_batch[j] for j in valid_indices]
                
                if not valid_queries:
                    # All queries in batch are empty
                    batch_results = [
                        TopicPrediction(
                            text=query,
                            predicted_topic=self.config.fallback_topic,
                            confidence=0.0
                        ) for query in batch
                    ]
                    results.extend(batch_results)
                    continue
                
                # Get predictions for valid queries
                predictions = self.classifier(valid_queries, top_k=1)
                
                # Process results
                batch_results = []
                valid_idx = 0
                
                for j, original_query in enumerate(batch):
                    if j in valid_indices:
                        # Valid query - use prediction
                        pred = predictions[valid_idx]
                        predicted_topic = self._validate_topic(pred[0]["label"])
                        confidence = float(pred[0]["score"])
                        
                        if confidence < self.config.confidence_threshold:
                            predicted_topic = self.config.fallback_topic
                        
                        batch_results.append(TopicPrediction(
                            text=original_query,
                            predicted_topic=predicted_topic,
                            confidence=confidence
                        ))
                        valid_idx += 1
                    else:
                        # Empty query - use fallback
                        batch_results.append(TopicPrediction(
                            text=original_query,
                            predicted_topic=self.config.fallback_topic,
                            confidence=0.0
                        ))
                
                results.extend(batch_results)
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i//self.config.batch_size + 1}: {e}")
                # Add fallback results for failed batch
                fallback_results = [
                    TopicPrediction(
                        text=query,
                        predicted_topic=self.config.fallback_topic,
                        confidence=0.0
                    ) for query in batch
                ]
                results.extend(fallback_results)
        
        total_time = time.time() - total_start_time
        self.logger.info(
            f"Classified {len(queries)} queries in {total_time:.2f}s "
            f"({len(queries)/total_time:.1f} queries/sec)"
        )
        
        return results
    
    def classify_query_variants(self, query_variants: Dict[str, List[str]]) -> Dict[str, List[TopicPrediction]]:
        """
        Classify query variants grouped by query_id.
        
        Args:
            query_variants: Dictionary mapping query_id to list of query variants
            
        Returns:
            Dictionary mapping query_id to list of TopicPrediction objects
        """
        self.logger.info(f"Classifying variants for {len(query_variants)} unique queries")
        
        results = {}
        
        for query_id, variants in tqdm(query_variants.items(), desc="Processing query variants"):
            results[query_id] = self.classify_batch(variants)
        
        return results
    
    def get_topic_statistics(self, predictions: List[TopicPrediction]) -> Dict[str, any]:
        """
        Get statistics about topic predictions.
        
        Args:
            predictions: List of TopicPrediction objects
            
        Returns:
            Dictionary with statistics
        """
        if not predictions:
            return {"total_predictions": 0}
        
        topics = [p.predicted_topic for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        from collections import Counter
        topic_counts = Counter(topics)
        
        stats = {
            "total_predictions": len(predictions),
            "unique_topics": len(topic_counts),
            "most_common_topics": topic_counts.most_common(5),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "low_confidence_count": sum(1 for c in confidences if c < self.config.confidence_threshold)
        }
        
        return stats


# Convenience functions
def classify_queries(queries: List[str], config: Optional[TopicClassifierConfig] = None) -> List[TopicPrediction]:
    """
    Convenience function to classify a list of queries.
    
    Args:
        queries: List of query texts
        config: Optional configuration
        
    Returns:
        List of TopicPrediction objects
    """
    classifier = TopicClassifier(config)
    return classifier.classify_batch(queries)


if __name__ == "__main__":
    # Test the topic classifier
    test_queries = [
        "Foreign film about strangers living in an apartment",
        "Horror movie with an old lady ghost",
        "Comedy about dads going down water slide",
        "Fantasy movie with giant computer",
        "Woman loses mother, ghost saves her"
    ]
    
    try:
        print("=== Testing Topic Classifier ===")
        config = TopicClassifierConfig(batch_size=8)
        classifier = TopicClassifier(config)
        
        # Test single classification
        print("\n--- Single Classification ---")
        result = classifier.classify_single(test_queries[0])
        print(f"Query: {result.text[:50]}...")
        print(f"Topic: {result.predicted_topic}")
        print(f"Confidence: {result.confidence:.3f}")
        
        # Test batch classification
        print("\n--- Batch Classification ---")
        results = classifier.classify_batch(test_queries)
        
        for result in results:
            print(f"Topic: {result.predicted_topic:20} "
                  f"Conf: {result.confidence:.3f} "
                  f"Query: {result.text[:40]}...")
        
        # Show statistics
        print("\n--- Statistics ---")
        stats = classifier.get_topic_statistics(results)
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error testing topic classifier: {e}")