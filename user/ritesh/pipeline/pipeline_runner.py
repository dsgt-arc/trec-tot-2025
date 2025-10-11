"""
Main Pipeline Runner for TREC-ToT 2025

This module orchestrates the complete pipeline from query input to TREC output:
1. Load query variants from JSONL
2. Classify topics using ModernBERT
3. Retrieve using FAISS multi-index
4. Process retrieval results (top 1000, sorted by score)
5. Format to TREC output
"""

import logging
import time
import argparse
from typing import Dict

from config import PipelineConfig, PresetConfigs
from data_loader import DataLoader, QueryData
from topic_classifier import TopicClassifier, TopicClassifierConfig
from retrieval_engine import FAISSMultiIndexRetriever, RetrievalConfig
from rank_fusion import RRFEngine, RRFConfig
from trec_formatter import TRECFormatter, TRECFormatterConfig

# Configure logging
def setup_logging(config: PipelineConfig):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if not config.enable_logging:
        logging.disable(logging.CRITICAL)


class TRECToTPipeline:
    """
    Main pipeline orchestrator for TREC-ToT 2025.
    
    Coordinates all components to process queries from input to TREC output.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")
        
        # Data loader
        self.data_loader = DataLoader()
        
        # Topic classifier
        topic_config = TopicClassifierConfig(
            batch_size=self.config.topic_batch_size,
            confidence_threshold=self.config.topic_confidence_threshold,
            fallback_topic=self.config.fallback_topic
        )
        self.topic_classifier = TopicClassifier(topic_config)
        
        # Retrieval engine
        retrieval_config = RetrievalConfig(
            faiss_indexes_dir=self.config.faiss_indexes_dir,
            embedding_model=self.config.embedding_model,
            results_per_variant=self.config.results_per_variant,
            use_gpu=self.config.use_gpu,
            batch_size_embedding=self.config.batch_size_embedding,
            index_cache_size=self.config.index_cache_size
        )
        self.retrieval_engine = FAISSMultiIndexRetriever(retrieval_config)
        
        # RRF fusion engine - not used in this version (direct retrieval processing)
        # rrf_config = RRFConfig(
        #     rrf_k=self.config.rrf_k,
        #     top_k_final=self.config.final_top_k,
        #     variant_weights=self.config.variant_weights,
        #     min_score_threshold=self.config.min_score_threshold
        # )
        # self.rrf_engine = RRFEngine(rrf_config)
        
        # TREC formatter
        trec_config = TRECFormatterConfig(
            run_id=self.config.run_id,
            max_results_per_query=self.config.final_top_k,
            score_precision=self.config.score_precision,
            overwrite_existing=self.config.overwrite_existing
        )
        self.trec_formatter = TRECFormatter(trec_config)
        
        self.logger.info("All components initialized successfully")
    
    def run(self) -> Dict:
        """
        Run the complete pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        
        self.logger.info("=" * 50)
        self.logger.info("Starting TREC-ToT 2025 Pipeline")
        self.logger.info("=" * 50)
        
        try:
            # Stage 1: Load and group query variants
            self.logger.info("Stage 1: Loading query variants...")
            
            grouped_queries = self.data_loader.load_and_group_queries(
                self.config.input_queries_file,
                self.config.expected_variants_per_query
            )
            
            self.logger.info(f"Loaded {len(grouped_queries)} unique query_ids")
            
            # Stage 2: Topic classification
            self.logger.info("Stage 2: Classifying query topics...")
            
            # Prepare query variants for classification
            query_variants_for_classification = {}
            for query_id, query_data in grouped_queries.items():
                query_variants_for_classification[query_id] = query_data.variants
            
            topic_results = self.topic_classifier.classify_query_variants(query_variants_for_classification)
            
            # Extract topic predictions
            topic_predictions = {}
            for query_id, predictions in topic_results.items():
                topic_predictions[query_id] = [pred.predicted_topic for pred in predictions]
            
            total_predictions = sum(len(preds) for preds in topic_results.values())
            self.logger.info(f"Classified {total_predictions} query variants")
            
            # Stage 3: Multi-index retrieval
            self.logger.info("Stage 3: Retrieving from FAISS indexes...")
            
            # Prepare data for retrieval
            retrieval_batch = {}
            for query_id in grouped_queries.keys():
                query_variants = query_variants_for_classification[query_id]
                topics = topic_predictions[query_id]
                retrieval_batch[query_id] = (query_variants, topics)
            
            retrieval_results = self.retrieval_engine.search_batch(retrieval_batch)
            
            total_results = sum(len(qr.results) for qr in retrieval_results.values())
            self.logger.info(f"Retrieved {total_results} total results")
            
            # Stage 4: Process retrieval results (top 1000, sorted by score)
            self.logger.info("Stage 4: Processing retrieval results (top 1000, sorted by score)...")
            
            # Convert retrieval results to TREC-compatible format
            processed_results = {}
            for query_id, retrieval_result in retrieval_results.items():
                # Get unique results (deduplicated by corpus_id, keeping best scores) and top 1000
                top_results = retrieval_result.get_unique_results(top_k=1000)
                
                # Convert to FusedResult format (mimicking RRF output structure)
                from rank_fusion import FusedResult, RRFResults
                fused_results = []
                for idx, result in enumerate(top_results, 1):
                    fused_result = FusedResult(
                        corpus_id=result.corpus_id,
                        fused_score=result.score,
                        final_rank=idx,
                        original_scores=[result.score],
                        original_ranks=[idx],
                        contributing_variants=[result.query_variant],
                        num_variants_present=1
                    )
                    fused_results.append(fused_result)
                
                # Create RRFResults-compatible structure
                processed_results[query_id] = RRFResults(
                    query_id=query_id,
                    fused_results=fused_results,
                    total_input_results=len(retrieval_result.results),
                    total_unique_docs=len(top_results),
                    variants_processed=retrieval_result.total_variants,
                    processing_time=retrieval_result.processing_time,
                    rrf_k_parameter=0,  # Not using RRF
                    avg_variants_per_doc=1.0,  # Since we're not doing fusion
                    docs_in_all_variants=0,
                    docs_in_single_variant=len(top_results)
                )
            
            total_processed = sum(len(result.fused_results) for result in processed_results.values())
            self.logger.info(f"Processed {len(processed_results)} queries, top results: {total_processed}")
            
            # Stage 5: TREC formatting
            self.logger.info("Stage 5: Formatting to TREC output...")
            
            trec_output = self.trec_formatter.format_batch(processed_results, self.config.output_trec_file)
            
            self.logger.info(f"Generated TREC file with {trec_output.total_entries} entries")
            
            # Optional JSONL output
            if self.config.output_jsonl_file:
                self.logger.info("Converting to JSONL format...")
                self.trec_formatter.convert_to_jsonl(
                    self.config.output_trec_file, 
                    self.config.output_jsonl_file
                )
                self.logger.info("JSONL file generated")
            
            # Validate output if requested
            if self.config.validate_output:
                self.logger.info("Validating TREC output...")
                validation = self.trec_formatter.validate_trec_file(self.config.output_trec_file)
                if validation['is_valid']:
                    self.logger.info("✓ TREC output validation passed")
                else:
                    self.logger.warning(f"✗ TREC output validation failed with {validation['error_count']} errors")
            
            total_time = time.time() - start_time
            
            self.logger.info("=" * 50)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Total execution time: {total_time:.2f}s")
            self.logger.info(f"Output file: {self.config.output_trec_file}")
            self.logger.info("=" * 50)
            
            return {
                "success": True,
                "total_time": total_time,
                "queries_processed": len(grouped_queries),
                "trec_entries": trec_output.total_entries,
                "output_file": self.config.output_trec_file
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            # Clean up resources
            if hasattr(self, 'retrieval_engine'):
                self.retrieval_engine.clear_cache()


# Convenience functions
def run_pipeline(config: PipelineConfig) -> Dict:
    """
    Run the complete pipeline with given configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Pipeline execution summary
    """
    setup_logging(config)
    pipeline = TRECToTPipeline(config)
    return pipeline.run()


def run_with_preset(preset_name: str = "dev_sample") -> Dict:
    """
    Run pipeline with a preset configuration.
    
    Args:
        preset_name: Name of preset configuration
        
    Returns:
        Pipeline execution summary
    """
    
    presets = {
        "dev_sample": PresetConfigs.dev_sample,
        "dev1_full": PresetConfigs.dev1_full,
        "test_set": PresetConfigs.test_set
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    config = presets[preset_name]()
    return run_pipeline(config)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="TREC-ToT 2025 Pipeline Runner")
    parser.add_argument(
        "--preset",
        choices=["dev_sample", "dev1_full", "test_set"],
        default="test_set",
        help="Preset configuration to use"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input queries file (overrides preset)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output TREC file (overrides preset)"
    )
    
    args = parser.parse_args()
    
    try:
        # Get preset config
        presets = {
            "dev_sample": PresetConfigs.dev_sample,
            "dev1_full": PresetConfigs.dev1_full,
            "test_set": PresetConfigs.test_set
        }
        
        config = presets[args.preset]()
        
        # Override with command line arguments
        if args.input:
            config.input_queries_file = args.input
        if args.output:
            config.output_trec_file = args.output
        
        # Run pipeline
        print(f"Running TREC-ToT 2025 Pipeline with preset: {args.preset}")
        summary = run_pipeline(config)
        
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED")
        print("=" * 50)
        print(f"Time: {summary['total_time']:.2f}s")
        print(f"Queries: {summary['queries_processed']}")
        print(f"TREC entries: {summary['trec_entries']}")
        print(f"Output: {summary['output_file']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        exit(1)