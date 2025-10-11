"""
Configuration Module for TREC-ToT 2025 Pipeline

This module contains all configuration settings for the complete pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class PipelineConfig:
    """Main configuration for the complete TREC-ToT 2025 pipeline."""
    
    # Input/Output paths
    input_queries_file: str = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/relaxed_queries_dev1-2025-queries_sample.jsonl"
    output_trec_file: str = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/results/multi_topic_fusion_results.trec"
    output_jsonl_file: Optional[str] = None  # Optional JSONL output
    faiss_indexes_dir: str = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/faiss_indexes"
    
    # Pipeline settings
    expected_variants_per_query: int = 6
    results_per_variant: int = 1000
    final_top_k: int = 1000
    run_id: str = "multi_topic_fusion"
    
    # Topic Classification settings
    topic_model: str = "davanstrien/ModernBERT-web-topics-1m"
    topic_batch_size: int = 32
    topic_confidence_threshold: float = 0.1
    fallback_topic: str = "entertainment"
    
    # Retrieval settings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    max_length: int = 512
    use_gpu: bool = True
    batch_size_embedding: int = 32
    index_cache_size: int = 3
    
    # RRF settings
    rrf_k: int = 60
    variant_weights: Optional[Dict[int, float]] = None
    min_score_threshold: float = 0.0
    
    # TREC formatting settings
    score_precision: int = 6
    validate_output: bool = True
    overwrite_existing: bool = True
    
    # Processing settings
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_statistics: bool = True
    save_intermediate_results: bool = False
    intermediate_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create output directories
        Path(self.output_trec_file).parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_jsonl_file:
            Path(self.output_jsonl_file).parent.mkdir(parents=True, exist_ok=True)
        
        if self.save_intermediate_results and self.intermediate_dir:
            Path(self.intermediate_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate paths
        if not Path(self.input_queries_file).exists():
            raise FileNotFoundError(f"Input queries file not found: {self.input_queries_file}")
        
        if not Path(self.faiss_indexes_dir).exists():
            raise FileNotFoundError(f"FAISS indexes directory not found: {self.faiss_indexes_dir}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


# Predefined configurations for different scenarios
class PresetConfigs:
    """Predefined configurations for common use cases."""
    
    @staticmethod
    def dev_sample() -> PipelineConfig:
        """Configuration for development/testing with sample data."""
        return PipelineConfig(
            input_queries_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/relaxed_queries_dev1-2025-queries_sample.jsonl",
            output_trec_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/results/sample_results.trec",
            final_top_k=100,  # Smaller for testing
            topic_batch_size=16,
            batch_size_embedding=16,
            index_cache_size=2,
            enable_logging=True,
            save_intermediate_results=True,
            intermediate_dir="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/intermediate"
        )
    
    @staticmethod
    def dev1_full() -> PipelineConfig:
        """Configuration for full dev1 dataset."""
        return PipelineConfig(
            input_queries_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/dev1-2025-queries-simplified_gemini.jsonl",
            output_trec_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/results/dev1_multi_topic_fusion_gemini2.trec",
            output_jsonl_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/results/dev1_multi_topic_fusion_gemini2.jsonl",
            final_top_k=1000,
            topic_batch_size=32,
            batch_size_embedding=32,
            index_cache_size=3,
            enable_statistics=True,
            save_intermediate_results=False
        )
    
    @staticmethod
    def test_set() -> PipelineConfig:
        """Configuration for test set runs."""
        return PipelineConfig(
            input_queries_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/relaxed-test-2025-queries_gemini.jsonl",
            output_trec_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/results/test-2025-dense_output.trec.trec",
            output_jsonl_file="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/results/test_multi_topic_fusion.jsonl",
            final_top_k=1000,
            topic_batch_size=64,
            batch_size_embedding=64,
            index_cache_size=5,
            enable_logging=True,
            log_level="WARNING",  # Less verbose
            save_intermediate_results=False,
            validate_output=True
        )