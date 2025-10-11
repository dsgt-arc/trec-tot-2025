"""
Updated build_and_save_faiss_index.py for IP + Normalization (Cosine Similarity)
"""

# Script to build and save a FAISS index for each topic from parquet files
import os
import pyarrow.parquet as pq
import numpy as np
import faiss

# Directory containing your parquet files
parquet_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet_cleaned"
index_out_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/faiss_indexes"
ids_out_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/faiss_indexes"
os.makedirs(index_out_dir, exist_ok=True)

embedding_dim = 1024  # BGE-M3 embedding dimension

# List of topics - focusing on entertainment first for testing
topics = [
    "entertainment"  # Start with just entertainment for testing
]

# Full list for later:
topics = [
    "adult_content","art_design","crime_law","education_jobs","electronics_hardware",
    "entertainment","fashion_beauty","finance_business","food_dining","games","health",
    "history_geography","home_hobbies","industrial","literature","politics","religion",
    "science_math_technology","social_life","software","software_development",
    "sports_fitness","transportation","travel_tourism"
]

def normalize_embeddings(embeddings):
    """Normalize embeddings for cosine similarity via inner product."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Add small epsilon to avoid division by zero
    norms = np.where(norms == 0, 1e-12, norms)
    return embeddings / norms

for topic in topics:
    parquet_file = os.path.join(parquet_dir, f"{topic}_cleaned_emb_bge-m3.parquet")
    if not os.path.exists(parquet_file):
        print(f"Parquet file for topic '{topic}' not found, skipping.")
        continue
    
    print(f"Processing topic: {topic}")
    
    # Load embeddings and IDs
    table = pq.read_table(parquet_file)
    emb = np.stack(table["embedding"].to_numpy())
    ids = table["id"].to_numpy()
    
    print(f"Loaded {len(emb)} embeddings for topic '{topic}'")
    
    # Convert to float32 and normalize for cosine similarity
    embeddings = emb.astype("float32")
    print(f"Before normalization - sample norms: {np.linalg.norm(embeddings[:5], axis=1)}")
    
    embeddings = normalize_embeddings(embeddings)
    print(f"After normalization - sample norms: {np.linalg.norm(embeddings[:5], axis=1)}")
    
    # Build FAISS Inner Product index (for cosine similarity with normalized vectors)
    try:
        if faiss.get_num_gpus() > 0:
            print(f"Using GPU for index building...")
            res = faiss.StandardGpuResources()
            # Create CPU index first, then transfer to GPU
            cpu_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            
            # Add embeddings to GPU index
            gpu_index.add(embeddings)
            
            # Transfer back to CPU for saving
            final_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            print(f"Using CPU for index building...")
            final_index = faiss.IndexFlatIP(embedding_dim)
            final_index.add(embeddings)
        
        print(f"Index built with {final_index.ntotal} vectors")
        
        # Save index and ids
        index_out_path = os.path.join(index_out_dir, f"faiss_index_{topic}.index")
        ids_out_path = os.path.join(ids_out_dir, f"ids_{topic}.npy")
        
        # Backup old index if it exists
        if os.path.exists(index_out_path):
            backup_path = index_out_path + ".backup"
            os.rename(index_out_path, backup_path)
            print(f"Backed up old index to {backup_path}")
        
        faiss.write_index(final_index, index_out_path)
        np.save(ids_out_path, np.array(ids))
        
        print(f"✅ Saved FAISS IP index to {index_out_path}")
        print(f"✅ Saved IDs to {ids_out_path}")
        
        # Quick verification
        test_query = np.random.random((1, embedding_dim)).astype(np.float32)
        test_query = normalize_embeddings(test_query)  # Normalize test query too
        
        scores, indices = final_index.search(test_query, 5)
        print(f"✅ Index verification - sample scores: {scores[0][:3]}")
        
    except Exception as e:
        print(f"❌ Error building index for topic '{topic}': {e}")
        continue

print("✅ Done building indexes!")
print("\n🔧 Next steps:")
print("1. Update retrieval_engine.py to use normalized queries")
print("2. Test with entertainment queries")
print("3. If successful, rebuild all topic indexes")