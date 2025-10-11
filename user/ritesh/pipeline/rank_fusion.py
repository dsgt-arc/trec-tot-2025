"""
Rank Fusion Module for TREC-ToT 2025 Pipeline

This module implements Reciprocal Rank Fusion (RRF) to combine ~6000 retrieval results
from 6 query variants into the final top-1000 ranked list.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Available rank fusion strategies."""
    RRF = "reciprocal_rank_fusion"
    COMBSUM = "comb_sum"
    COMBMNZ = "comb_mnz"


@dataclass
class FusedResult:
    """Represents a result after rank fusion."""
    corpus_id: str
    fused_score: float
    final_rank: int
    
    # RRF provenance
    original_scores: List[float]         # Original scores from each contributing variant
    original_ranks: List[int]           # Original ranks from each contributing variant  
    contributing_variants: List[int]     # Which variants contributed this result
    num_variants_present: int           # How many variants contained this document
    
    def to_dict(self) -> Dict:
        return {
            'corpus_id': self.corpus_id,
            'fused_score': self.fused_score,
            'final_rank': self.final_rank,
            'num_variants_present': self.num_variants_present,
            'original_scores': self.original_scores,
            'original_ranks': self.original_ranks,
            'contributing_variants': self.contributing_variants
        }


@dataclass
class RRFResults:
    """Results of RRF fusion process."""
    query_id: str
    fused_results: List[FusedResult]    # Top-1000 results after RRF
    total_input_results: int            # ~6000 total input results
    total_unique_docs: int              # Unique corpus_ids after RRF
    variants_processed: int             # Should be 6
    processing_time: float
    rrf_k_parameter: int
    
    # Statistics
    avg_variants_per_doc: float         # Average variants per unique document
    docs_in_all_variants: int          # Documents appearing in all 6 variants
    docs_in_single_variant: int        # Documents appearing in only 1 variant
    
    def get_top_k(self, k: int) -> List[FusedResult]:
        """Get top-k fused results."""
        return self.fused_results[:k]
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get detailed RRF statistics."""
        if not self.fused_results:
            return {"total_results": 0}
        
        variant_counts = [r.num_variants_present for r in self.fused_results]
        scores = [r.fused_score for r in self.fused_results]
        
        return {
            "final_results_count": len(self.fused_results),
            "total_input_results": self.total_input_results,
            "total_unique_docs": self.total_unique_docs,
            "deduplication_ratio": 1.0 - (len(self.fused_results) / self.total_input_results) if self.total_input_results > 0 else 0.0,
            "variants_processed": self.variants_processed,
            "avg_variants_per_doc": self.avg_variants_per_doc,
            "docs_in_all_variants": self.docs_in_all_variants,
            "docs_in_single_variant": self.docs_in_single_variant,
            "min_rrf_score": min(scores) if scores else 0.0,
            "max_rrf_score": max(scores) if scores else 0.0,
            "avg_rrf_score": sum(scores) / len(scores) if scores else 0.0,
            "processing_time": self.processing_time,
            "rrf_k_parameter": self.rrf_k_parameter
        }


class RRFConfig:
    """Configuration for RRF fusion."""
    
    def __init__(
        self,
        rrf_k: int = 60,                              # Standard RRF parameter
        top_k_final: int = 1000,                     # Final number of results to return
        variant_weights: Optional[Dict[int, float]] = None,  # Optional weights for variants
        min_score_threshold: float = 0.0             # Minimum score threshold
    ):
        self.rrf_k = rrf_k
        self.top_k_final = top_k_final
        self.variant_weights = variant_weights or {}
        self.min_score_threshold = min_score_threshold


class RRFEngine:
    """
    Reciprocal Rank Fusion engine for combining ~6000 results into top-1000.
    
    Takes 6 ranked lists (one per query variant) and applies RRF formula:
    RRF_score(doc) = Σ(1/(k + rank_in_variant_i)) for all variants containing doc
    """
    
    def __init__(self, config: Optional[RRFConfig] = None):
        self.config = config or RRFConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fuse_query_variants(
        self, 
        variant_results: List[List], 
        query_id: str = "unknown"
    ) -> RRFResults:
        """
        Apply RRF to fuse results from 6 query variants.
        
        Args:
            variant_results: List of 6 ranked lists, each with ~1000 RetrievalResult objects
            query_id: Query identifier for tracking
            
        Returns:
            RRFResults with top-1000 fused results
        """
        import time
        start_time = time.time()
        
        if not variant_results or not any(variant_results):
            return self._empty_results(query_id, time.time() - start_time)
        
        # Count total input results
        total_input_results = sum(len(vr) for vr in variant_results)
        self.logger.debug(f"Processing {total_input_results} total results from {len(variant_results)} variants")
        
        # Group results by corpus_id and calculate RRF scores
        corpus_data = defaultdict(lambda: {
            "rrf_score": 0.0,
            "variants": [],
            "scores": [],
            "ranks": []
        })
        
        # Process each variant
        for variant_idx, variant_result_list in enumerate(variant_results):
            variant_weight = self.config.variant_weights.get(variant_idx, 1.0)
            
            for rank, result in enumerate(variant_result_list):
                corpus_id = result.corpus_id
                original_score = result.best_score if hasattr(result, 'best_score') else result.score
                
                # Skip if below threshold
                if original_score < self.config.min_score_threshold:
                    continue
                
                # Calculate RRF contribution from this variant
                rrf_contribution = variant_weight / (self.config.rrf_k + rank + 1)  # rank is 0-based, make it 1-based
                
                # Accumulate data
                corpus_data[corpus_id]["rrf_score"] += rrf_contribution
                corpus_data[corpus_id]["variants"].append(variant_idx)
                corpus_data[corpus_id]["scores"].append(original_score)
                corpus_data[corpus_id]["ranks"].append(rank + 1)  # Store as 1-based rank
        
        # Convert to FusedResult objects
        fused_results = []
        for corpus_id, data in corpus_data.items():
            fused_result = FusedResult(
                corpus_id=corpus_id,
                fused_score=data["rrf_score"],
                final_rank=0,  # Will be set after sorting
                original_scores=data["scores"],
                original_ranks=data["ranks"],
                contributing_variants=data["variants"],
                num_variants_present=len(data["variants"])
            )
            fused_results.append(fused_result)
        
        # Sort by RRF score (descending) and assign final ranks
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)
        
        # Take top-k and assign final ranks
        final_results = fused_results[:self.config.top_k_final]
        for rank, result in enumerate(final_results):
            result.final_rank = rank + 1
        
        # Calculate statistics - FIXED: Use data from corpus_data, not fused objects
        variant_counts = [len(data["variants"]) for data in corpus_data.values()]
        num_variants = len(variant_results)
        
        avg_variants_per_doc = sum(variant_counts) / len(variant_counts) if variant_counts else 0.0
        docs_in_all_variants = sum(1 for count in variant_counts if count == num_variants)
        docs_in_single_variant = sum(1 for count in variant_counts if count == 1)
        
        processing_time = time.time() - start_time
        
        self.logger.info(
            f"RRF: {total_input_results} → {len(corpus_data)} unique → {len(final_results)} final "
            f"(k={self.config.rrf_k}) in {processing_time:.3f}s"
        )
        
        return RRFResults(
            query_id=query_id,
            fused_results=final_results,
            total_input_results=total_input_results,
            total_unique_docs=len(corpus_data),
            variants_processed=len(variant_results),
            processing_time=processing_time,
            rrf_k_parameter=self.config.rrf_k,
            avg_variants_per_doc=avg_variants_per_doc,
            docs_in_all_variants=docs_in_all_variants,
            docs_in_single_variant=docs_in_single_variant
        )
    
    def fuse_batch(
        self, 
        batch_results: Dict[str, List[List]]
    ) -> Dict[str, RRFResults]:
        """
        Apply RRF to multiple queries.
        
        Args:
            batch_results: Dict mapping query_id to list of 6 variant result lists
            
        Returns:
            Dict mapping query_id to RRFResults
        """
        fused_batch = {}
        
        self.logger.info(f"Processing {len(batch_results)} queries with RRF")
        
        for query_id, variant_results in batch_results.items():
            fused_batch[query_id] = self.fuse_query_variants(variant_results, query_id)
        
        return fused_batch
    
    def _empty_results(self, query_id: str, processing_time: float) -> RRFResults:
        """Create empty RRFResults object."""
        return RRFResults(
            query_id=query_id,
            fused_results=[],
            total_input_results=0,
            total_unique_docs=0,
            variants_processed=0,
            processing_time=processing_time,
            rrf_k_parameter=self.config.rrf_k,
            avg_variants_per_doc=0.0,
            docs_in_all_variants=0,
            docs_in_single_variant=0
        )
    
    def get_batch_statistics(self, batch_results: Dict[str, RRFResults]) -> Dict[str, Union[int, float]]:
        """Get statistics for batch RRF processing."""
        if not batch_results:
            return {"total_queries": 0}
        
        total_input = sum(r.total_input_results for r in batch_results.values())
        total_unique = sum(r.total_unique_docs for r in batch_results.values())
        total_final = sum(len(r.fused_results) for r in batch_results.values())
        processing_times = [r.processing_time for r in batch_results.values()]
        
        return {
            "total_queries": len(batch_results),
            "total_input_results": total_input,
            "total_unique_docs": total_unique,
            "total_final_results": total_final,
            "avg_input_per_query": total_input / len(batch_results) if batch_results else 0.0,
            "avg_unique_per_query": total_unique / len(batch_results) if batch_results else 0.0,
            "avg_final_per_query": total_final / len(batch_results) if batch_results else 0.0,
            "overall_deduplication_ratio": 1.0 - (total_final / total_input) if total_input > 0 else 0.0,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "total_processing_time": sum(processing_times)
        }


# Convenience function
def apply_rrf_fusion(
    variant_results: List[List],
    query_id: str = "unknown",
    rrf_k: int = 60,
    top_k: int = 1000
) -> RRFResults:
    """
    Convenience function to apply RRF to query variants.
    
    Args:
        variant_results: List of 6 ranked lists from query variants
        query_id: Query identifier
        rrf_k: RRF parameter (default 60 is standard)
        top_k: Number of final results (default 1000)
        
    Returns:
        RRFResults with top-k fused results
    """
    config = RRFConfig(rrf_k=rrf_k, top_k_final=top_k)
    rrf_engine = RRFEngine(config)
    return rrf_engine.fuse_query_variants(variant_results, query_id)


if __name__ == "__main__":
    # Test RRF with realistic scale
    print("=== Testing RRF Engine ===")
    
    # Create mock results simulating 6 variants with ~1000 results each
    def create_mock_variant(variant_id: int, num_docs: int = 1000) -> List:
        results = []
        for i in range(num_docs):
            # Create some overlap between variants (simulate real scenario)
            if variant_id == 0:
                corpus_id = f"doc_{i}"
            else:
                # Create ~70% overlap with previous variants
                if i < 700:
                    corpus_id = f"doc_{i}"
                else:
                    corpus_id = f"doc_{variant_id}_{i}"
            
            score = 1.0 - (i / num_docs) + (variant_id * 0.01)  # Decreasing scores with slight variant bias
            
            # Mock result object
            result = type('MockResult', (), {
                'corpus_id': corpus_id,
                'score': score,
                'best_score': score
            })()
            results.append(result)
        
        return results
    
    # Create test data: 6 variants with ~1000 results each
    test_variants = [create_mock_variant(i, 1000) for i in range(6)]
    total_input = sum(len(v) for v in test_variants)
    
    print(f"Created test data: {len(test_variants)} variants, {total_input} total results")
    
    try:
        # Test RRF
        config = RRFConfig(rrf_k=60, top_k_final=1000)
        rrf_engine = RRFEngine(config)
        
        rrf_results = rrf_engine.fuse_query_variants(test_variants, "test_query")
        
        # Show results
        print(f"\n--- RRF Results ---")
        print(f"Input results: {rrf_results.total_input_results}")
        print(f"Unique documents: {rrf_results.total_unique_docs}")
        print(f"Final top-1000: {len(rrf_results.fused_results)}")
        print(f"Deduplication ratio: {(1 - len(rrf_results.fused_results)/rrf_results.total_input_results):.2%}")
        print(f"Processing time: {rrf_results.processing_time:.3f}s")
        print(f"Avg variants per doc: {rrf_results.avg_variants_per_doc:.2f}")
        print(f"Docs in all 6 variants: {rrf_results.docs_in_all_variants}")
        print(f"Docs in single variant: {rrf_results.docs_in_single_variant}")
        
        # Show top 10 results
        print(f"\nTop 10 RRF Results:")
        for result in rrf_results.get_top_k(10):
            print(f"  {result.final_rank}: {result.corpus_id} "
                  f"(RRF: {result.fused_score:.4f}, variants: {result.num_variants_present})")
        
        # Test batch processing
        print(f"\n--- Testing Batch Processing ---")
        batch_data = {
            "query_1": test_variants,
            "query_2": [create_mock_variant(i, 800) for i in range(6)]  # Smaller second query
        }
        
        batch_results = rrf_engine.fuse_batch(batch_data)
        batch_stats = rrf_engine.get_batch_statistics(batch_results)
        
        print("Batch Statistics:")
        for key, value in batch_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error testing RRF: {e}")
        import traceback
        traceback.print_exc()