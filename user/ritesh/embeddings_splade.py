# embeddings_splade.py
import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import pickle
from collections import defaultdict
import json


def build_splade_model(model_name="naver/splade_v2_max", device=None, use_fp16=True):
    """
    Build SPLADE model similar to build_model() in BGE script.
    
    Args:
        model_name: SPLADE model from HuggingFace
        device: Device to load on (auto-detect if None)
        use_fp16: Whether to use half precision
    
    Returns:
        Tuple of (tokenizer, model)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SPLADE model: {model_name} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    model.to(device)
    if use_fp16 and device == "cuda":
        model = model.half()
    
    model.eval()
    
    print(f"✅ SPLADE model loaded: {model_name}")
    return tokenizer, model


def doc_to_chunks_splade(text, tokenizer, max_len=512, stride=256, min_tokens=16):
    """
    Convert a long document into overlapping token windows for SPLADE.
    Similar to doc_to_chunks() in BGE script.
    """
    # Get token ids once (no truncation)
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(ids) <= max_len:
        return [text]  # short doc -> single chunk

    chunks_ids = []
    step = max(1, max_len - stride)
    for start in range(0, len(ids), step):
        window = ids[start:start + max_len]
        if len(window) < min_tokens:
            break  # tiny tail
        chunks_ids.append(window)
        if start + max_len >= len(ids):
            break

    # Decode all windows at once (faster than per-window decode)
    chunks = tokenizer.batch_decode(chunks_ids, skip_special_tokens=True)
    return chunks if chunks else [text]


def encode_texts_splade(tokenizer, model, texts, batch_size=32, max_length=512, 
                       splade_threshold=0.01, show_progress=True):
    """
    Encode a list of texts to SPLADE sparse vectors.
    Fixed version that handles progress bar correctly.
    """
    device = next(model.parameters()).device
    sparse_vectors = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Create range iterator with optional progress bar
    if show_progress:
        range_iterator = tqdm(range(num_batches), desc="SPLADE encoding", unit="batch")
    else:
        range_iterator = range(num_batches)
    
    # Process in batches
    for batch_idx in range_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch = texts[start_idx:end_idx]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            # Get SPLADE logits
            outputs = model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Apply SPLADE transformations
            sparse_weights = torch.log1p(torch.relu(logits))
            
            # Max pooling over sequence dimension
            doc_reps = torch.max(sparse_weights, dim=1)[0]  # [batch_size, vocab_size]
            
            # Convert to sparse dictionaries
            for doc_rep in doc_reps:
                doc_rep = doc_rep.cpu()
                
                # Filter by threshold
                non_zero_indices = torch.nonzero(doc_rep > splade_threshold).squeeze(-1)
                
                if len(non_zero_indices) > 0:
                    sparse_dict = {
                        str(idx.item()): doc_rep[idx].item()
                        for idx in non_zero_indices
                    }
                else:
                    sparse_dict = {}
                
                sparse_vectors.append(sparse_dict)
        
        # Clear GPU memory periodically
        if device == "cuda" and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return sparse_vectors


def embed_docs_with_chunking_splade(tokenizer, model, titles, texts, batch_size=32, 
                                   max_len=512, stride=256, splade_threshold=0.01):
    """
    For each doc (title + text), create chunks by tokens, embed each chunk with SPLADE,
    then aggregate chunk sparse vectors to a single sparse vector per doc.
    Similar to embed_docs_with_chunking() in BGE script.
    
    Returns:
        List of sparse vectors (each is dict {term_id: weight})
    """
    # 1) Build chunk lists per doc
    doc_chunks = []
    doc_chunk_counts = []
    
    print("Creating document chunks...")
    for title, body in tqdm(zip(titles, texts), total=len(titles)):
        # Simple concat; keep title to boost topical signal
        full = f"{title} {body}" if isinstance(title, str) else (body or "")
        chunks = doc_to_chunks_splade(full, tokenizer, max_len=max_len, stride=stride)
        doc_chunks.append(chunks)
        doc_chunk_counts.append(len(chunks))

    # 2) Flatten chunks and embed in batches
    all_chunks = []
    for chunks in doc_chunks:
        all_chunks.extend(chunks)
    
    print(f"Encoding {len(all_chunks)} chunks with SPLADE...")
    
    # Encode all chunks
    chunk_sparse_vectors = encode_texts_splade(
        tokenizer, model, all_chunks, 
        batch_size=batch_size, 
        max_length=max_len,
        splade_threshold=splade_threshold,
        show_progress=True
    )
    
    # 3) Aggregate chunks back to documents
    print("Aggregating chunks to documents...")
    doc_sparse_vectors = []
    chunk_idx = 0
    
    for num_chunks in doc_chunk_counts:
        if num_chunks == 1:
            # Single chunk - use as is
            doc_sparse_vectors.append(chunk_sparse_vectors[chunk_idx])
        else:
            # Multiple chunks - aggregate by max pooling weights
            aggregated_vector = defaultdict(float)
            
            for i in range(num_chunks):
                chunk_vector = chunk_sparse_vectors[chunk_idx + i]
                for term_id, weight in chunk_vector.items():
                    # Max pooling - keep highest weight for each term
                    aggregated_vector[term_id] = max(aggregated_vector[term_id], weight)
            
            # Convert back to dict and filter
            final_vector = {
                str(term_id): weight 
                for term_id, weight in aggregated_vector.items() 
                if weight > splade_threshold
            }
            doc_sparse_vectors.append(final_vector)
        
        chunk_idx += num_chunks
    
    return doc_sparse_vectors


def process_parquet_to_splade(
    input_file: str,
    output_file: str,
    model_name: str = "naver/splade_v2_max",
    batch_size: int = 32,
    max_length: int = 512,
    stride: int = 256,
    splade_threshold: float = 0.01,
    device: str = None,
    use_fp16: bool = True,
    text_column: str = "text",
    title_column: str = "title",
    id_column: str = "id"
):
    """
    Process parquet file to generate SPLADE sparse embeddings.
    Similar to main processing function in BGE script.
    
    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path  
        model_name: SPLADE model name
        batch_size: Batch size for encoding
        max_length: Max sequence length for chunks
        stride: Overlap between chunks
        splade_threshold: Minimum weight to keep in sparse vectors
        device: Device to use (auto-detect if None)
        use_fp16: Use half precision
        text_column: Name of text column in parquet
        title_column: Name of title column in parquet  
        id_column: Name of ID column in parquet
    """
    
    print("=" * 50)
    print("SPLADE Embedding Generation")
    print("=" * 50)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print(f"SPLADE threshold: {splade_threshold}")
    print("=" * 50)
    
    # Build model
    tokenizer, model = build_splade_model(model_name, device, use_fp16)
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} documents")
    
    # pick the fisrt 100 documents
    # df = df.head(100)

    # Extract columns
    texts = df[text_column].fillna("").astype(str).tolist()
    titles = df[title_column].fillna("").astype(str).tolist() if title_column in df.columns else [""] * len(texts)
    ids = df[id_column].tolist()
    
    print(f"Processing {len(texts)} documents...")
    
    # Generate SPLADE embeddings
    start_time = time.time()
    
    sparse_vectors = embed_docs_with_chunking_splade(
        tokenizer, model, titles, texts,
        batch_size=batch_size,
        max_len=max_length,
        stride=stride,
        splade_threshold=splade_threshold
    )
    
    encoding_time = time.time() - start_time
    print(f"Encoding completed in {encoding_time:.2f} seconds")
    print(f"Average: {encoding_time/len(texts):.4f} seconds per document")

    # 🔧 FIX: Serialize SPLADE vectors to JSON strings for safe storage
    print("Serializing SPLADE vectors for safe parquet storage...")
    
    serialized_sparse_vectors = []
    for sparse_vector in sparse_vectors:
        # Convert to JSON string
        json_vector = json.dumps(sparse_vector)
        serialized_sparse_vectors.append(json_vector)
    
    # Prepare output data with JSON serialized vectors
    output_data = {
        id_column: ids,
        "splade_vector_json": serialized_sparse_vectors,  # Store as JSON strings
        "num_terms": [len(vec) for vec in sparse_vectors],
        "max_weight": [max(vec.values()) if vec else 0.0 for vec in sparse_vectors],
        "total_weight": [sum(vec.values()) if vec else 0.0 for vec in sparse_vectors]
    }

    
    # Add original columns if needed
    if text_column in df.columns:
        output_data[text_column] = texts
    if title_column in df.columns and title_column in df.columns:
        output_data[title_column] = titles
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Save statistics
    print(f"SPLADE Statistics:")
    print(f"  Average terms per document: {output_df['num_terms'].mean():.1f}")
    print(f"  Max terms in a document: {output_df['num_terms'].max()}")
    print(f"  Documents with no terms: {(output_df['num_terms'] == 0).sum()}")
    print(f"  Average max weight: {output_df['max_weight'].mean():.4f}")
    print(f"  Average total weight: {output_df['total_weight'].mean():.4f}")
    
    # Save to parquet
    print(f"Saving to {output_file}...")
    output_df.to_parquet(output_file, compression='snappy')
    
    print(f"✅ SPLADE embeddings saved to {output_file}")
    print(f"✅ Total processing time: {time.time() - start_time:.2f} seconds")
    
    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return output_df


def batch_process_topics(
    input_dir: str,
    output_dir: str,
    topics: List[str],
    model_name: str = "naver/splade_v2_max",
    **kwargs
):
    """
    Process multiple topic parquet files.
    Similar to batch processing in BGE script.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for topic in topics:
        input_file = os.path.join(input_dir, f"{topic}_cleaned_emb_bge-m3.parquet")
        output_file = os.path.join(output_dir, f"{topic}_splade.parquet")
        
        if not os.path.exists(input_file):
            print(f"⚠️  Input file not found: {input_file}")
            continue
        
        if os.path.exists(output_file):
            print(f"⚠️  Output file exists, skipping: {output_file}")
            continue
        
        print(f"\n🚀 Processing topic: {topic}")
        
        try:
            process_parquet_to_splade(
                input_file=input_file,
                output_file=output_file,
                model_name=model_name,
                **kwargs
            )
            print(f"✅ Completed: {topic}")
            
        except Exception as e:
            print(f"❌ Error processing {topic}: {e}")
            continue


def create_splade_search_index(
    parquet_file: str,
    index_output_path: str,
    splade_threshold: float = 0.01
):
    """
    Create searchable SPLADE index from parquet file.
    Optimized for fast retrieval.
    
    Args:
        parquet_file: Input parquet with SPLADE vectors
        index_output_path: Where to save the search index
        splade_threshold: Minimum weight threshold
    """
    
    print(f"Creating SPLADE search index from {parquet_file}...")
    
    # Load data
    df = pd.read_parquet(parquet_file)
    corpus_ids = df['id'].astype(str).tolist()
    
    # 🔧 FIX: Deserialize JSON strings back to dictionaries
    print("Deserializing SPLADE vectors from JSON...")
    splade_vectors = []
    for json_vector in df['splade_vector_json']:
        try:
            sparse_vector = json.loads(json_vector)
            splade_vectors.append(sparse_vector)
        except Exception as e:
            print(f"⚠️  Error deserializing vector: {e}")
            splade_vectors.append({})  # Empty vector as fallback
    
    # Create inverted index: term_id -> [(doc_idx, weight), ...]
    print("Building inverted index...")
    inverted_index = defaultdict(list)
    
    for doc_idx, sparse_vector in enumerate(tqdm(splade_vectors, desc="Indexing")):
        for term_id, weight in sparse_vector.items():
            if weight is None:
                continue
            if weight > splade_threshold:
                inverted_index[term_id].append((doc_idx, weight))
    
    # Sort postings by weight (descending)
    for term_id in inverted_index:
        inverted_index[term_id].sort(key=lambda x: x[1], reverse=True)
    
    # Save search index
    search_index = {
        'inverted_index': dict(inverted_index),
        'corpus_ids': corpus_ids,
        'vocab_size': len(inverted_index),
        'num_docs': len(corpus_ids),
        'threshold': splade_threshold
    }
    
    with open(index_output_path, 'wb') as f:
        pickle.dump(search_index, f)
    
    print(f"✅ SPLADE search index saved: {index_output_path}")
    print(f"✅ Vocabulary size: {len(inverted_index)}")
    print(f"✅ Documents indexed: {len(corpus_ids)}")
    
    return search_index

def check_splade_model_capacity(model_name="naver/splade_v2_max"):
    """Check what the SPLADE model can actually handle."""
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    
    print(f"SPLADE Model Analysis: {model_name}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Vocab size: {config.vocab_size}")
    
    return config.max_position_embeddings

class SPLADESearcher:
    """
    Fast SPLADE searcher using inverted index.
    Similar interface to your existing retrievers.
    """
    
    def __init__(self, index_path: str, model_name: str = "naver/splade_v2_max"):
        """Initialize SPLADE searcher."""
        
        # Load search index
        print(f"Loading SPLADE search index from {index_path}...")
        with open(index_path, 'rb') as f:
            self.index_data = pickle.load(f)
        
        self.inverted_index = self.index_data['inverted_index']
        self.corpus_ids = self.index_data['corpus_ids']
        
        # Load SPLADE model for query encoding
        self.tokenizer, self.model = build_splade_model(model_name)
        
        print(f"✅ SPLADE searcher ready")
        print(f"  Vocabulary: {self.index_data['vocab_size']} terms")
        print(f"  Documents: {self.index_data['num_docs']} docs")
    
    def encode_query(self, query: str, splade_threshold: float = 0.01) -> Dict[int, float]:
        """Encode query to SPLADE sparse vector."""
        sparse_vectors = encode_texts_splade(
            self.tokenizer, self.model, [query],
            batch_size=1,
            splade_threshold=splade_threshold,
            show_progress=False
        )
        return sparse_vectors[0]
    
    def search(self, query: str, top_k: int = 1000, splade_threshold: float = 0.01):
        """
        Search using SPLADE sparse vectors.
        
        Args:
            query: Query string
            top_k: Number of results to return
            splade_threshold: Threshold for query terms
            
        Returns:
            List of dicts with corpus_id, score, rank
        """
        
        # Encode query
        query_vector = self.encode_query(query, splade_threshold)
        
        if not query_vector:
            return []
        
        # Score documents using inverted index
        doc_scores = defaultdict(float)
        
        for term_id, query_weight in query_vector.items():
            if term_id in self.inverted_index:
                # Add scores for all documents containing this term
                for doc_idx, doc_weight in self.inverted_index[term_id]:
                    doc_scores[doc_idx] += query_weight * doc_weight
        
        # Sort by score and get top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Convert to results
        results = []
        for rank, (doc_idx, score) in enumerate(sorted_docs):
            if score > 0:
                result = {
                    'corpus_id': self.corpus_ids[doc_idx],
                    'score': score,
                    'rank': rank + 1
                }
                results.append(result)
        
        return results


if __name__ == "__main__":

    topic = "history_geography"
    max_supported_length = check_splade_model_capacity(model_name="naver/splade_v2_max")
    print(f"Max supported length: {max_supported_length}")
    # Configuration (easily modifiable)
    CONFIG = {
        # Model settings
        "model_name": "naver/splade_v2_max",  # Change this to try different models
        "device": None,  # Auto-detect
        "use_fp16": True,
        
        # Processing settings  
        "batch_size": 16,  # Reduce if memory issues
        "max_length": max_supported_length,
        "stride": max_supported_length // 2,
        "splade_threshold": 0.01,  # Lower = more terms kept
        
        # File paths
        "input_file": f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/cleaned_articles_parquet/{topic}_cleaned.parquet",
        "output_file": f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/splade_embeddings/{topic}_splade.parquet",
        "index_file": f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/splade_indexes/{topic}_splade.pkl"
    }
    
    # Create output directories
    os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["index_file"]), exist_ok=True)
    
    # Step 1: Generate SPLADE embeddings
    print("Step 1: Generating SPLADE embeddings...")
    df = process_parquet_to_splade(
        input_file=CONFIG["input_file"],
        output_file=CONFIG["output_file"],
        model_name=CONFIG["model_name"],
        batch_size=CONFIG["batch_size"],
        max_length=CONFIG["max_length"],
        stride=CONFIG["stride"],
        splade_threshold=CONFIG["splade_threshold"],
        device=CONFIG["device"],
        use_fp16=CONFIG["use_fp16"]
    )
    
    # Step 2: Create search index
    print("\nStep 2: Creating search index...")
    search_index = create_splade_search_index(
        parquet_file=CONFIG["output_file"],
        index_output_path=CONFIG["index_file"],
        splade_threshold=CONFIG["splade_threshold"]
    )
    
    # Step 3: Quick test
    print("\nStep 3: Testing searcher...")
    searcher = SPLADESearcher(CONFIG["index_file"], CONFIG["model_name"])
    
    test_queries = [
        "Taylor Swift pop singer",
        "Marvel superhero movie",
        "comedy film funny"
    ]
    
    for query in test_queries:
        print(f"\nTest query: {query}")
        results = searcher.search(query, top_k=5)
        
        for result in results[:3]:
            print(f"  Score: {result['score']:.3f} | Doc: {result['corpus_id']}")
    
    print(f"\n✅ SPLADE setup complete!")
    print(f"✅ Ready for integration with your pipeline")