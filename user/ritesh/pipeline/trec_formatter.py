"""
TREC Formatter Module for TREC-ToT 2025 Pipeline

This module converts RRF fusion results to standard TREC format output files.
Format: <query_id> Q0 <corpus_id> <rank> <score> <run_id>
"""

import logging
import os
from typing import Dict, List, Optional, TextIO, Union
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TRECEntry:
    """Represents a single TREC format entry."""
    query_id: str
    q0: str = "Q0"              # Always "Q0" in TREC format
    corpus_id: str = ""
    rank: int = 0
    score: float = 0.0
    run_id: str = ""
    
    def to_trec_line(self) -> str:
        """Convert to TREC format line."""
        return f"{self.query_id} {self.q0} {self.corpus_id} {self.rank} {self.score:.6f} {self.run_id}"
    
    def __str__(self) -> str:
        return self.to_trec_line()


@dataclass
class TRECOutput:
    """Results of TREC formatting."""
    output_file: str
    total_entries: int
    unique_queries: int
    entries_per_query: Dict[str, int]
    processing_time: float
    run_id: str
    
    def get_statistics(self) -> Dict[str, Union[int, float, str]]:
        """Get TREC output statistics."""
        avg_entries = sum(self.entries_per_query.values()) / len(self.entries_per_query) if self.entries_per_query else 0.0
        
        return {
            "output_file": self.output_file,
            "total_entries": self.total_entries,
            "unique_queries": self.unique_queries,
            "avg_entries_per_query": avg_entries,
            "max_entries_per_query": max(self.entries_per_query.values()) if self.entries_per_query else 0,
            "min_entries_per_query": min(self.entries_per_query.values()) if self.entries_per_query else 0,
            "processing_time": self.processing_time,
            "run_id": self.run_id,
            "file_size_mb": os.path.getsize(self.output_file) / (1024 * 1024) if os.path.exists(self.output_file) else 0.0
        }


class TRECFormatterConfig:
    """Configuration for TREC formatting."""
    
    def __init__(
        self,
        run_id: str = "multi_topic_fusion",
        max_results_per_query: int = 1000,
        score_precision: int = 6,
        sort_by_score: bool = True,
        validate_format: bool = True,
        overwrite_existing: bool = True
    ):
        self.run_id = run_id
        self.max_results_per_query = max_results_per_query
        self.score_precision = score_precision
        self.sort_by_score = sort_by_score
        self.validate_format = validate_format
        self.overwrite_existing = overwrite_existing


class TRECFormatter:
    """
    Formats RRF fusion results into standard TREC format files.
    
    TREC Format: <query_id> Q0 <corpus_id> <rank> <score> <run_id>
    """
    
    def __init__(self, config: Optional[TRECFormatterConfig] = None):
        self.config = config or TRECFormatterConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _validate_query_id(self, query_id: str) -> bool:
        """Validate query ID format."""
        if not query_id or not isinstance(query_id, str):
            return False
        
        # Remove common problematic characters
        if any(char in query_id for char in [' ', '\t', '\n', '\r']):
            return False
        
        return True
    
    def _validate_corpus_id(self, corpus_id: str) -> bool:
        """Validate corpus ID format."""
        if not corpus_id or not isinstance(corpus_id, str):
            return False
        
        # Remove common problematic characters
        if any(char in corpus_id for char in [' ', '\t', '\n', '\r']):
            return False
        
        return True
    
    def _create_trec_entries(self, query_id: str, rrf_results) -> List[TRECEntry]:
        """
        Create TREC entries from RRF results for a single query.
        
        Args:
            query_id: Query identifier
            rrf_results: RRFResults object with fused results
            
        Returns:
            List of TRECEntry objects
        """
        entries = []
        
        # Validate query ID
        if not self._validate_query_id(query_id):
            self.logger.error(f"Invalid query ID: '{query_id}'")
            return entries
        
        # Get results (already sorted by RRF score in descending order)
        results = rrf_results.get_top_k(self.config.max_results_per_query)
        
        for result in results:
            # Validate corpus ID
            if not self._validate_corpus_id(result.corpus_id):
                self.logger.warning(f"Skipping invalid corpus ID: '{result.corpus_id}' for query {query_id}")
                continue
            
            # Create TREC entry
            entry = TRECEntry(
                query_id=query_id,
                corpus_id=result.corpus_id,
                rank=result.final_rank,
                score=result.fused_score,
                run_id=self.config.run_id
            )
            entries.append(entry)
        
        return entries
    
    def format_single_query(
        self, 
        query_id: str, 
        rrf_results, 
        output_file: Optional[str] = None
    ) -> List[TRECEntry]:
        """
        Format results for a single query to TREC format.
        
        Args:
            query_id: Query identifier
            rrf_results: RRFResults object
            output_file: Optional file to write to
            
        Returns:
            List of TRECEntry objects
        """
        entries = self._create_trec_entries(query_id, rrf_results)
        
        # Write to file if specified
        if output_file:
            self._write_entries_to_file(entries, output_file, mode='w')
        
        self.logger.debug(f"Formatted {len(entries)} entries for query {query_id}")
        return entries
    
    def format_batch(
        self, 
        batch_results: Dict[str, any], 
        output_file: str
    ) -> TRECOutput:
        """
        Format batch of RRF results to TREC format file.
        
        Args:
            batch_results: Dictionary mapping query_id to RRFResults objects
            output_file: Path to output TREC file
            
        Returns:
            TRECOutput object with formatting results
        """
        import time
        start_time = time.time()
        
        if not batch_results:
            return self._empty_output(output_file, time.time() - start_time)
        
        # Prepare output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists() and not self.config.overwrite_existing:
            raise FileExistsError(f"Output file exists and overwrite_existing=False: {output_file}")
        
        all_entries = []
        entries_per_query = {}
        
        # Process each query
        for query_id, rrf_results in batch_results.items():
            entries = self._create_trec_entries(query_id, rrf_results)
            all_entries.extend(entries)
            entries_per_query[query_id] = len(entries)
        
        # Sort entries by query_id, then by rank
        all_entries.sort(key=lambda x: (x.query_id, x.rank))
        
        # Write to file
        self._write_entries_to_file(all_entries, str(output_path), mode='w')
        
        processing_time = time.time() - start_time
        
        self.logger.info(
            f"Formatted {len(all_entries)} entries for {len(batch_results)} queries "
            f"to {output_file} in {processing_time:.3f}s"
        )
        
        return TRECOutput(
            output_file=str(output_path),
            total_entries=len(all_entries),
            unique_queries=len(batch_results),
            entries_per_query=entries_per_query,
            processing_time=processing_time,
            run_id=self.config.run_id
        )
    
    def _write_entries_to_file(
        self, 
        entries: List[TRECEntry], 
        output_file: str, 
        mode: str = 'w'
    ):
        """Write TREC entries to file."""
        try:
            with open(output_file, mode, encoding='utf-8') as f:
                for entry in entries:
                    f.write(entry.to_trec_line() + '\n')
        except Exception as e:
            self.logger.error(f"Error writing to {output_file}: {e}")
            raise
    
    def _empty_output(self, output_file: str, processing_time: float) -> TRECOutput:
        """Create empty TRECOutput object."""
        return TRECOutput(
            output_file=output_file,
            total_entries=0,
            unique_queries=0,
            entries_per_query={},
            processing_time=processing_time,
            run_id=self.config.run_id
        )
    
    def validate_trec_file(self, trec_file: str) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate TREC format file.
        
        Args:
            trec_file: Path to TREC file
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        line_count = 0
        query_ids = set()
        
        try:
            with open(trec_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # Parse TREC line
                    parts = line.split()
                    if len(parts) != 6:
                        errors.append(f"Line {line_num}: Expected 6 fields, got {len(parts)}")
                        continue
                    
                    query_id, q0, corpus_id, rank, score, run_id = parts
                    
                    # Validate format
                    if q0 != "Q0":
                        errors.append(f"Line {line_num}: Second field should be 'Q0', got '{q0}'")
                    
                    try:
                        rank_int = int(rank)
                        if rank_int <= 0:
                            errors.append(f"Line {line_num}: Rank should be positive, got {rank_int}")
                    except ValueError:
                        errors.append(f"Line {line_num}: Invalid rank '{rank}'")
                    
                    try:
                        float(score)
                    except ValueError:
                        errors.append(f"Line {line_num}: Invalid score '{score}'")
                    
                    query_ids.add(query_id)
        
        except Exception as e:
            errors.append(f"Error reading file: {e}")
        
        is_valid = len(errors) == 0
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "line_count": line_count,
            "unique_queries": len(query_ids),
            "error_count": len(errors)
        }
    
    def convert_to_jsonl(
        self, 
        trec_file: str, 
        jsonl_file: str
    ) -> Dict[str, int]:
        """
        Convert TREC format file to JSONL format.
        
        Args:
            trec_file: Input TREC file
            jsonl_file: Output JSONL file
            
        Returns:
            Conversion statistics
        """
        import json
        
        converted_count = 0
        
        try:
            with open(trec_file, 'r', encoding='utf-8') as infile, \
                 open(jsonl_file, 'w', encoding='utf-8') as outfile:
                
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 6:
                        continue
                    
                    query_id, q0, corpus_id, rank, score, run_id = parts
                    
                    entry = {
                        "query_id": query_id,
                        "corpus_id": corpus_id,
                        "rank": int(rank),
                        "score": float(score),
                        "run_id": run_id
                    }
                    
                    outfile.write(json.dumps(entry) + '\n')
                    converted_count += 1
        
        except Exception as e:
            self.logger.error(f"Error converting {trec_file} to {jsonl_file}: {e}")
            raise
        
        return {"converted_entries": converted_count}


# Convenience functions
def format_to_trec(
    batch_results: Dict[str, any],
    output_file: str,
    run_id: str = "multi_topic_fusion",
    max_results: int = 1000
) -> TRECOutput:
    """
    Convenience function to format RRF results to TREC file.
    
    Args:
        batch_results: Dictionary mapping query_id to RRFResults objects
        output_file: Path to output TREC file
        run_id: Run identifier for TREC format
        max_results: Maximum results per query
        
    Returns:
        TRECOutput object
    """
    config = TRECFormatterConfig(run_id=run_id, max_results_per_query=max_results)
    formatter = TRECFormatter(config)
    return formatter.format_batch(batch_results, output_file)


if __name__ == "__main__":
    # Test the TREC formatter
    from rank_fusion import FusedResult, RRFResults
    
    # Create mock RRF results
    mock_fused_results = [
        FusedResult("doc1", 0.95, 1, [0.9, 0.8], [1, 2], [0, 1], 2),
        FusedResult("doc2", 0.88, 2, [0.85], [1], [0], 1),
        FusedResult("doc3", 0.82, 3, [0.8, 0.75, 0.78], [2, 3, 1], [0, 1, 2], 3),
        FusedResult("doc4", 0.75, 4, [0.7, 0.72], [4, 3], [1, 2], 2),
    ]
    
    mock_rrf_results = RRFResults(
        query_id="test_152",
        fused_results=mock_fused_results,
        total_input_results=6000,
        total_unique_docs=1500,
        variants_processed=6,
        processing_time=0.125,
        rrf_k_parameter=60,
        avg_variants_per_doc=2.3,
        docs_in_all_variants=50,
        docs_in_single_variant=800
    )
    
    test_batch = {
        "152": mock_rrf_results,
        "531": mock_rrf_results  # Reuse for testing
    }
    
    try:
        print("=== Testing TREC Formatter ===")
        
        # Test formatting
        config = TRECFormatterConfig(run_id="test_run", max_results_per_query=10)
        formatter = TRECFormatter(config)
        
        output_file = "/tmp/test_results.trec"
        trec_output = formatter.format_batch(test_batch, output_file)
        
        print(f"TREC Output Statistics:")
        stats = trec_output.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Show file contents
        print(f"\nFirst 10 lines of {output_file}:")
        try:
            with open(output_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"  Error reading file: {e}")
        
        # Test validation
        print(f"\n--- File Validation ---")
        validation = formatter.validate_trec_file(output_file)
        print(f"Valid: {validation['is_valid']}")
        print(f"Lines: {validation['line_count']}")
        print(f"Unique queries: {validation['unique_queries']}")
        if validation['errors']:
            print(f"Errors: {validation['error_count']}")
            for error in validation['errors'][:5]:
                print(f"  {error}")
        
        # Test JSONL conversion
        print(f"\n--- JSONL Conversion ---")
        jsonl_file = "/tmp/test_results.jsonl"
        conversion_stats = formatter.convert_to_jsonl(output_file, jsonl_file)
        print(f"Converted entries: {conversion_stats['converted_entries']}")
        
    except Exception as e:
        print(f"Error testing TREC formatter: {e}")
        import traceback
        traceback.print_exc()