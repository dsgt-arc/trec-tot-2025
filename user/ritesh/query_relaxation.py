"""
Query Relaxation using Llama 3 8B Instruct Model

This module provides functionality to relax and expand queries using the Llama 3 8B Instruct model.
Query relaxation helps improve search performance by generating alternative query formulations
that capture different aspects of the original query.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RelaxedQuery:
    """Represents a relaxed query with its metadata."""
    original_query: str
    relaxed_query: str
    relaxation_type: str
    confidence_score: float
    reasoning: str


class QueryRelaxationConfig:
    """Configuration for query relaxation."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_relaxations: int = 3,
        batch_size: int = 32,
        device: Optional[str] = None,
        use_4bit: bool = True,
        use_flash_attention: bool = False
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.num_relaxations = num_relaxations
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_4bit = use_4bit
        self.use_flash_attention = use_flash_attention


class QueryRelaxationEngine:
    """
    Engine for relaxing queries using Llama 3 8B Instruct model.
    
    This class provides methods to generate relaxed versions of queries that can
    help improve search performance by capturing different aspects and formulations
    of the original query.
    """
    
    def __init__(self, config: QueryRelaxationConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Llama 3 8B Instruct model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            model_kwargs = {
                "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
                "device_map": "auto" if self.config.device == "cuda" else None,
                "trust_remote_code": True,
            }
            
            if self.config.use_4bit and self.config.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, using full precision")
            
            if self.config.use_flash_attention and self.config.device == "cuda":
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2")
                except ImportError:
                    logger.warning("flash-attn not available, using standard attention")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            if self.config.device == "cpu":
                self.model = self.model.to(self.config.device)
            
            logger.info(f"Model loaded successfully on {self.config.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _create_prompt(self, query: str, relaxation_type: str) -> str:
        """Create a prompt for query relaxation."""
        
        prompts = {
            "relaxed": f"""You are helping retrieve documents from a large knowledge base.
The user query may contain incorrect or overly specific details.
You need to help improve search recall by generating multiple variants of the query.
Only modify terms that could block recall if wrong.  
Do not add any facts that are not in the original query.
Do not change the core subject.

TASK:
- Keep the main meaning.
- In Variant 1: Keep all key terms but remove one uncertain detail.
- In Variant 2: Remove multiple uncertain details, keep core nouns/verbs.
- In Variant 3: Keep only the essential topic and generalize specific terms.

Query: {query}

Return exactly 3 lines and nothing else.
Each line must start with "Q: " followed by the relaxed query (≤ 40 words).
""",
"semantic": f"""You are preparing queries for a search system.

For the given query:
1. Keep the topic and intent.
2. Replace rare or niche words with common synonyms or broader terms.
3. Output 3 alternative queries, each slightly different.
4. Do not add facts not present in the query.
5. Keep at least one noun from the original in each version.

Query: {query}

Return exactly 3 lines and nothing else.
Each line must start with "Q: " followed by the relaxed query (≤ 40 words).
"""
        }

        return prompts.get(relaxation_type, prompts["relaxed"])
    
    def _generate_relaxations(self, query: str, relaxation_type: str) -> List[str]:
        """Generate relaxed queries using the model."""
        
        prompt = self._create_prompt(query, relaxation_type)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract relaxed queries from response
        relaxed_queries = self._extract_relaxed_queries(response, relaxation_type)
        
        return relaxed_queries[:self.config.num_relaxations]
    
    def _extract_relaxed_queries(self, response: str, relaxation_type: str) -> List[str]:
        """Extract relaxed queries from the model response."""
        queries = []
        
        # Look for lines starting with "Q: "
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and line.startswith('Q: '):
                # remove everything after the first full stop
                line = line.split('.', 1)[0]
                queries.append(line[3:])
    
        return queries
    
    def relax_query(self, query: str, relaxation_types: Optional[List[str]] = None) -> List[str]:
        """
        Relax a single query using multiple relaxation strategies.
        
        Args:
            query: The original query to relax
            relaxation_types: List of relaxation types to apply. If None, uses all available types.
        
        Returns:
            List of relaxed queries
        """
        if relaxation_types is None:
            relaxation_types = ["semantic", "relaxed"]
        
        relaxed_queries = []
        
        for relaxation_type in relaxation_types:
            try:
                logger.info(f"Generating {relaxation_type} relaxations for query")
                
                relaxed_texts = self._generate_relaxations(query, relaxation_type)
                
                # for i, relaxed_text in enumerate(relaxed_texts):
                #     relaxed_query = RelaxedQuery(
                #         original_query=query,
                #         relaxed_query=relaxed_text,
                #         relaxation_type=relaxation_type,
                #         confidence_score=0.8 - (i * 0.1),  # Simple confidence scoring
                #         reasoning=f"Generated using {relaxation_type} strategy"
                #     )
                relaxed_queries.extend(relaxed_texts)
                
                # Add small delay to avoid overwhelming the model
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating {relaxation_type} relaxations: {e}")
                continue
        
        return relaxed_queries
    
    def relax_queries_batch(self, queries: List[str], relaxation_types: Optional[List[str]] = None) -> List[List[RelaxedQuery]]:
        """
        Relax multiple queries in batch.
        
        Args:
            queries: List of original queries
            relaxation_types: List of relaxation types to apply
        
        Returns:
            List of lists of RelaxedQuery objects, one list per input query
        """
        results = []
        
        for query_data in tqdm(queries, desc="Relaxing queries"):
            query_id = query_data["query_id"]
            query_text = query_data["query"]
            
            relaxed_queries = self.relax_query(query_text, relaxation_types)

            # all_queries = [query_text] + relaxed_queries
            all_queries = relaxed_queries
            for query in all_queries:
                results.append({
                    "query_id": query_id,
                    "query": query
                })
        
        return results
    
    def save_relaxations(self, relaxations: List[List[str]], output_file: str):
        """Save relaxed queries to a JSONL file."""
        
        # Save to file in JSONL format (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for query_data in relaxations:
                # Write each query as a separate line in JSONL format
                json.dump(query_data, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved relaxations to {output_file}")


def load_queries_from_jsonl(file_path: str) -> List[str]:
    """Load queries from a JSONL file."""
    queries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries.append({
                    "query_id": data["query_id"],
                    "query": data["query"]
                })
    
    return queries


def main():
    """Main function to demonstrate query relaxation."""

    from huggingface_hub import login
    login(token="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")

    # Configuration
    config = QueryRelaxationConfig(
        # model_name="Qwen/Qwen2.5-32B-Instruct",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        num_relaxations=3,
        temperature=0.7,
        max_length=512
    )
    
    # Initialize the relaxation engine
    engine = QueryRelaxationEngine(config)
    
    # Load test queries
    name = "dev1-2025-queries"
    queries_file = f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/{name}.jsonl"
    if os.path.exists(queries_file):
        queries = load_queries_from_jsonl(queries_file)
        logger.info(f"Loaded {len(queries)} queries from {queries_file}")
        
        # Take a small sample for testing
        sample_queries = queries  # Process first 5 queries
        
        # Generate relaxations
        relaxations = engine.relax_queries_batch(sample_queries)
        
        # Save results
        output_file = f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/relaxed_queries_{name}.jsonl"
        engine.save_relaxations(relaxations, output_file)
    
    else:
        # Test with a single example query
        test_query = "horror movie with a old lady , possibly a ghost killing in an old house . .\n This is an older 80 s movie. Maybe early 90s. I remember a old lady doing the killing .she may be a ghost , also a guy gets killed out side on a ladder in the rain at night. I also remember a long haired stoner kind of guy that makes a weapon with a saw blade. The house that this happens in is having work done and the guy that makes the weapon is a carpenter ."
        
        relaxations = engine.relax_query(test_query)

        print(relaxations)


if __name__ == "__main__":
    main()
