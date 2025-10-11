"""
Multi-Index FAISS Retrieval Engine for TREC-ToT 2025 Pipeline

This module handles FAISS-based retrieval across multiple topic-specific indexes
using BGE-M3 embeddings. Supports batch processing and efficient index management.
"""

import os
import gc
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import faiss
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    corpus_id: str
    score: float
    faiss_index_id: int
    topic: str
    query_variant: int
    
    def to_dict(self) -> Dict:
        return {
            'corpus_id': self.corpus_id,
            'score': self.score,
            'faiss_index_id': self.faiss_index_id,
            'topic': self.topic,
            'query_variant': self.query_variant
        }


@dataclass
class QueryRetrievalResults:
    """Results for all variants of a single query."""
    query_id: str
    results: List[RetrievalResult]
    total_variants: int
    unique_corpus_ids: int
    processing_time: float
    
    def get_unique_results(self, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Get unique results (deduplicated by corpus_id), keeping best scores."""
        unique_results = {}
        for result in self.results:
            if result.corpus_id not in unique_results:
                unique_results[result.corpus_id] = result
            else:
                # Keep the result with higher score
                if result.score > unique_results[result.corpus_id].score:
                    unique_results[result.corpus_id] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k] if top_k else sorted_results


class RetrievalConfig:
    """Configuration for FAISS retrieval."""
    
    def __init__(
        self,
        faiss_indexes_dir: str = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/faiss_indexes",
        embedding_model: str = "BAAI/bge-m3",
        embedding_dim: int = 1024,
        max_length: int = 512,
        results_per_variant: int = 1000,
        device: Optional[Union[str, int]] = None,
        use_gpu: bool = True,
        batch_size_embedding: int = 32,
        index_cache_size: int = 1  # Number of indexes to keep in memory
    ):
        self.faiss_indexes_dir = Path(faiss_indexes_dir)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.results_per_variant = results_per_variant
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size_embedding = batch_size_embedding
        self.index_cache_size = index_cache_size


class FAISSMultiIndexRetriever:
    """
    Multi-index FAISS retrieval engine with topic-specific indexes.
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize embedding model
        self.embedding_model = None
        self._initialize_embedding_model()
        
        # Index cache
        self.index_cache: Dict[str, Tuple[faiss.Index, np.ndarray]] = {}
        self.index_usage_order: List[str] = []
        
        # Validate available indexes
        self.available_topics = self._discover_available_topics()
    
    def _initialize_embedding_model(self):
        """Initialize BGE-M3 embedding model."""
        try:
            self.logger.info(f"Initializing embedding model: {self.config.embedding_model}")
            self.embedding_model = BGEM3FlagModel(
                self.config.embedding_model,
                use_fp16=True,
                device=self.config.device
            )
            self.logger.info(f"Embedding model initialized on device: {self.config.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _discover_available_topics(self) -> Dict[str, bool]:
        """Discover available topic indexes and validate them."""
        available = {}
        
        if not self.config.faiss_indexes_dir.exists():
            self.logger.error(f"FAISS indexes directory not found: {self.config.faiss_indexes_dir}")
            return available
        
        for index_file in self.config.faiss_indexes_dir.glob("faiss_index_*.index"):
            topic = index_file.stem.replace("faiss_index_", "")
            ids_file = self.config.faiss_indexes_dir / f"ids_{topic}.npy"
            
            if ids_file.exists():
                available[topic] = True
                self.logger.debug(f"Found complete index for topic: {topic}")
            else:
                available[topic] = False
                self.logger.warning(f"Missing IDs file for topic: {topic}")
        
        self.logger.info(f"Found {sum(available.values())} complete topic indexes")
        return available
    
    def _load_index_and_ids(self, topic: str) -> Tuple[faiss.Index, np.ndarray]:
        """Load FAISS index and corpus IDs for a topic."""
        if topic in self.index_cache:
            # Move to end of usage order (most recently used)
            self.index_usage_order.remove(topic)
            self.index_usage_order.append(topic)
            return self.index_cache[topic]
        
        # Load from disk
        index_path = self.config.faiss_indexes_dir / f"faiss_index_{topic}.index"
        ids_path = self.config.faiss_indexes_dir / f"ids_{topic}.npy"
        
        if not index_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Index or IDs file not found for topic: {topic}")
        
        try:
            # Load FAISS index
            self.logger.debug(f"Loading FAISS index for topic: {topic}")
            index = faiss.read_index(str(index_path))
            
            # Move to GPU if available and requested
            try:
                if self.config.use_gpu and faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    self.logger.debug(f"Moved index for topic {topic} to GPU")
            except:
                pass
            
            # Load corpus IDs
            corpus_ids = np.load(str(ids_path), allow_pickle=True)
            
            # Validate dimensions
            if index.ntotal != len(corpus_ids):
                raise ValueError(
                    f"Index size ({index.ntotal}) doesn't match corpus IDs size ({len(corpus_ids)}) "
                    f"for topic: {topic}"
                )
            
            # Cache management
            if len(self.index_cache) >= self.config.index_cache_size:
                # Remove least recently used index
                oldest_topic = self.index_usage_order.pop(0)
                del self.index_cache[oldest_topic]
                gc.collect()
                self.logger.debug(f"Evicted index for topic: {oldest_topic}")
            
            # Add to cache
            self.index_cache[topic] = (index, corpus_ids)
            self.index_usage_order.append(topic)
            
            self.logger.debug(f"Loaded and cached index for topic: {topic}")
            return index, corpus_ids
            
        except Exception as e:
            self.logger.error(f"Failed to load index for topic {topic}: {e}")
            raise
    
    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries to dense embeddings using BGE-M3."""
        try:

            # Add instruction prefix to all queries
            # processed_queries = [
            #     f"Represent this sentence for searching relevant passages: {query}" 
            #     for query in queries
            # ]

            # Use BGE-M3 dense encoding
            embeddings = self.embedding_model.encode(
                queries,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                batch_size=self.config.batch_size_embedding,
                max_length=self.config.max_length,
                convert_to_numpy=True
            )["dense_vecs"]
            
            # Normalize embeddings (important for cosine similarity)
            embeddings = embeddings.astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode queries: {e}")
            raise
    
    def search_single_topic(
        self, 
        queries: List[str], 
        topic: str, 
        top_k: Optional[int] = None
    ) -> List[List[RetrievalResult]]:
        """
        Search a single topic index with multiple queries.
        
        Args:
            queries: List of query strings
            topic: Topic name to search
            top_k: Number of results per query (defaults to config.results_per_variant)
            
        Returns:
            List of lists, where each inner list contains RetrievalResult objects for one query
        """
        if not queries:
            return []
        
        if topic not in self.available_topics or not self.available_topics[topic]:
            self.logger.warning(f"Topic {topic} not available, skipping")
            return [[] for _ in queries]
        
        top_k = top_k or self.config.results_per_variant
        
        try:
            # Load index and IDs
            index, corpus_ids = self._load_index_and_ids(topic)
            
            # Encode queries
            query_embeddings = self._encode_queries(queries)
            
            # Search
            scores, indices = index.search(query_embeddings, top_k)
            
            # Convert results
            results = []
            for query_idx in range(len(queries)):
                query_results = []
                for rank, (score, faiss_idx) in enumerate(zip(scores[query_idx], indices[query_idx])):
                    if faiss_idx >= 0 and faiss_idx < len(corpus_ids):  # Valid result
                        corpus_id = str(corpus_ids[faiss_idx])
                        query_results.append(RetrievalResult(
                            corpus_id=corpus_id,
                            score=float(score),
                            faiss_index_id=int(faiss_idx),
                            topic=topic,
                            query_variant=query_idx
                        ))
                results.append(query_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search topic {topic}: {e}")
            return [[] for _ in queries]
    
    def search_multi_topic(
        self,
        queries: List[str],
        topics: List[str],
        top_k: Optional[int] = None
    ) -> List[List[RetrievalResult]]:
        """
        Search multiple topics with the same queries.
        
        Args:
            queries: List of query strings
            topics: List of topic names to search
            top_k: Number of results per query per topic
            
        Returns:
            List of lists, where each inner list contains combined results for one query
        """
        if not queries or not topics:
            return []
        
        all_results = [[] for _ in queries]
        
        for topic in topics:
            topic_results = self.search_single_topic(queries, topic, top_k)
            
            # Merge results for each query
            for query_idx, query_topic_results in enumerate(topic_results):
                all_results[query_idx].extend(query_topic_results)
        
        # Sort results for each query by score
        for query_results in all_results:
            query_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results
    
    def search_query_variants(
        self,
        query_id: str,
        query_variants: List[str],
        topic_predictions: List[str]
    ) -> QueryRetrievalResults:
        """
        Search with multiple query variants and their predicted topics.
        
        Args:
            query_id: Unique query identifier
            query_variants: List of query variant strings
            topic_predictions: List of predicted topics (same length as query_variants)
            
        Returns:
            QueryRetrievalResults object with combined results
        """
        start_time = time.time()
        
        if len(query_variants) != len(topic_predictions):
            raise ValueError("Number of query variants must match number of topic predictions")
        
        # Group queries by topic for efficient processing
        topic_to_queries: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for variant_idx, (query, topic) in enumerate(zip(query_variants, topic_predictions)):
            topic_to_queries[topic].append((variant_idx, query))
        
        # Collect all results
        all_results = []
        
        # Process each topic group
        for topic, variant_queries in topic_to_queries.items():
            variant_indices = [vq[0] for vq in variant_queries]
            queries = [vq[1] for vq in variant_queries]
            
            # Search this topic
            topic_results = self.search_single_topic(queries, topic)
            
            # Add variant information to results
            for local_query_idx, variant_idx in enumerate(variant_indices):
                for result in topic_results[local_query_idx]:
                    result.query_variant = variant_idx
                    all_results.append(result)
        
        # Create final results object
        unique_corpus_ids = len(set(r.corpus_id for r in all_results))
        processing_time = time.time() - start_time
        
        return QueryRetrievalResults(
            query_id=query_id,
            results=all_results,
            total_variants=len(query_variants),
            unique_corpus_ids=unique_corpus_ids,
            processing_time=processing_time
        )
    
    def search_batch(
        self,
        batch_data: Dict[str, Tuple[List[str], List[str]]]
    ) -> Dict[str, QueryRetrievalResults]:
        """
        Search batch of queries with their variants and topic predictions.
        
        Args:
            batch_data: Dictionary mapping query_id to (query_variants, topic_predictions)
            
        Returns:
            Dictionary mapping query_id to QueryRetrievalResults
        """
        results = {}
        
        for query_id, (query_variants, topic_predictions) in tqdm(
            batch_data.items(), desc="Searching query variants"
        ):
            results[query_id] = self.search_query_variants(
                query_id, query_variants, topic_predictions
            )
        
        return results
    
    def get_retrieval_statistics(self, results: Dict[str, QueryRetrievalResults]) -> Dict[str, any]:
        """Get statistics about retrieval results."""
        if not results:
            return {"total_queries": 0}
        
        total_results = sum(len(qr.results) for qr in results.values())
        total_unique = sum(qr.unique_corpus_ids for qr in results.values())
        processing_times = [qr.processing_time for qr in results.values()]
        
        # Topic distribution
        topic_counts = defaultdict(int)
        for qr in results.values():
            for result in qr.results:
                topic_counts[result.topic] += 1
        
        stats = {
            "total_queries": len(results),
            "total_results": total_results,
            "total_unique_corpus_ids": total_unique,
            "avg_results_per_query": total_results / len(results) if results else 0,
            "avg_unique_per_query": total_unique / len(results) if results else 0,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "topic_distribution": dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)),
            "available_topics": len(self.available_topics)
        }
        
        return stats
    
    def clear_cache(self):
        """Clear the index cache to free memory."""
        self.index_cache.clear()
        self.index_usage_order.clear()
        gc.collect()
        self.logger.info("Cleared index cache")


# Convenience function
def search_queries(
    queries_with_topics: Dict[str, Tuple[List[str], List[str]]],
    config: Optional[RetrievalConfig] = None
) -> Dict[str, QueryRetrievalResults]:
    """
    Convenience function to search multiple queries with topic predictions.
    
    Args:
        queries_with_topics: Dict mapping query_id to (query_variants, topic_predictions)
        config: Optional retrieval configuration
        
    Returns:
        Dictionary mapping query_id to QueryRetrievalResults
    """
    retriever = FAISSMultiIndexRetriever(config)
    return retriever.search_batch(queries_with_topics)


if __name__ == "__main__":
    # Test the retrieval engine
    test_queries = {
        "test_query_1": (
            ["Foreign film about strangers in apartment", "Art house film with three people"],
            ["entertainment", "entertainment"]
        ),
        "test_query_2": (
            ["Horror movie with old lady ghost", "80s horror film in old house"],
            ["entertainment", "entertainment"]
        )
    }
    
    try:
        print("=== Testing FAISS Multi-Index Retriever ===")
        config = RetrievalConfig(results_per_variant=10)
        retriever = FAISSMultiIndexRetriever(config)
        
        # Show available topics
        print(f"\nAvailable topics: {len(retriever.available_topics)}")
        for topic, available in list(retriever.available_topics.items())[:5]:
            print(f"  {topic}: {'✓' if available else '✗'}")
        
        # Test search
        print("\n--- Testing Search ---")
        results = retriever.search_batch(test_queries)
        
        for query_id, qr_results in results.items():
            print(f"\nQuery: {query_id}")
            print(f"  Total results: {len(qr_results.results)}")
            print(f"  Unique corpus IDs: {qr_results.unique_corpus_ids}")
            print(f"  Processing time: {qr_results.processing_time:.3f}s")
            
            # Show top results
            unique_results = qr_results.get_unique_results(5)
            for i, result in enumerate(unique_results[:3]):
                print(f"    {i+1}: {result.corpus_id} (score: {result.score:.4f}, topic: {result.topic})")
        
        # Show statistics
        print("\n--- Statistics ---")
        stats = retriever.get_retrieval_statistics(results)
        for key, value in stats.items():
            if key != "topic_distribution":
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error testing retrieval engine: {e}")
        import traceback
        traceback.print_exc()