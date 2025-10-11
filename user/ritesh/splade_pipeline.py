# simple_splade_pipeline.py
import json
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
import numpy as np
from collections import defaultdict
import logging
import time

# Configuration
query_file = "relaxed-test-2025-queries_gemini1.jsonl"
trec_file = "test-2025-splade_output1.trec"
run_id = "splade_run"

CONFIG = {
    "input_queries": f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/{query_file}",
    "splade_model": "naver/splade_v2_max", 
    "topic_model": "davanstrien/ModernBERT-web-topics-1m",
    "splade_indexes": f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/splade_indexes/",
    "output_trec": f"/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/results/{trec_file}",
    "top_k": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 512,
    "AVAILABLE_TOPICS": [
        "adult_content", "art_design", "crime_law", "education_jobs",
        "electronics_hardware", "entertainment", "fashion_beauty", 
        "finance_business", "food_dining", "games", "health",
        "history_geography", "home_hobbies", "industrial", "literature",
        "politics", "religion", "science_math_technology", "social_life",
        "software", "software_development", "sports_fitness", 
        "transportation", "travel_tourism"],
    }

class SimpleSPLADEPipeline:
    def __init__(self, config):
        self.config = config
        self.load_models()
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        self.index_data = None
        self.inverted_index = None
        self.corpus_ids = None
        self.saved_index = {}
        self.topic_name = None
        self.entertainment_data = None  # Cache for entertainment index
        self.politics_data = None  # Cache for politics index
        self.sports_fitness_data = None  # Cache for sports_fitness index
        self.science_math_technology_data = None  # Cache for science_math_technology index
        
    def load_models(self):
        # Load SPLADE model
        self.splade_tokenizer = AutoTokenizer.from_pretrained(self.config["splade_model"])
        self.splade_model = AutoModelForMaskedLM.from_pretrained(self.config["splade_model"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.splade_model.to(device)
        self.splade_model.eval()
        
        # Load topic model  
        self.topic_model = pipeline(
                "text-classification",
                model=self.config["topic_model"],
                tokenizer=self.config["topic_model"],
                device=self.config["device"],
                truncation=True,
                max_length=self.config["max_length"]
            )
    
    def load_splade_index(self):
        # if self.topic_name in self.saved_index:
        #     self.index_data = self.saved_index[self.topic_name]
        # else:
        if self.topic_name == "entertainment":
            if self.entertainment_data is not None:
                self.index_data = self.entertainment_data
            else:
                with open(self.config["splade_indexes"] + self.topic_name + "_splade.pkl", 'rb') as f:
                    self.index_data = pickle.load(f)
                    self.entertainment_data = self.index_data
        # elif self.topic_name == "history_geography":
        #     if self.history_geography_data is not None:
        #         self.index_data = self.history_geography_data
        #     else:
        #         with open(self.config["splade_indexes"] + self.topic_name + "_splade.pkl", 'rb') as f:
        #             self.index_data = pickle.load(f)
        #             self.history_geography_data = self.index_data
        elif self.topic_name == "politics":
            if self.politics_data is not None:
                self.index_data = self.politics_data
            else:
                with open(self.config["splade_indexes"] + self.topic_name + "_splade.pkl", 'rb') as f:
                    self.index_data = pickle.load(f)
                    self.politics_data = self.index_data
        elif self.topic_name == "sports_fitness":
            if self.sports_fitness_data is not None:
                self.index_data = self.sports_fitness_data
            else:
                with open(self.config["splade_indexes"] + self.topic_name + "_splade.pkl", 'rb') as f:
                    self.index_data = pickle.load(f)
                    self.sports_fitness_data = self.index_data
        # elif self.topic_name == "science_math_technology":
        #     if self.science_math_technology_data is not None:
        #         self.index_data = self.science_math_technology_data
        #     else:
        #         with open(self.config["splade_indexes"] + self.topic_name + "_splade.pkl", 'rb') as f:
        #             self.index_data = pickle.load(f)
        #             self.science_math_technology_data = self.index_data
        else:
            with open(self.config["splade_indexes"] + self.topic_name + "_splade.pkl", 'rb') as f:
                self.index_data = pickle.load(f)
                # self.saved_index[self.topic_name] = self.index_data
                
        self.inverted_index = self.index_data['inverted_index']
        self.corpus_ids = self.index_data['corpus_ids']

    def classify_topic(self, query):
        predictions = self.topic_model(query, top_k=1)
        predicted_topic = predictions[0]["label"]
        
        # Normalize topic name
        normalized = predicted_topic.lower().split('includes')[0]
        
        # Try to find matches
        for topic in self.config["AVAILABLE_TOPICS"]:
            topic_parts = topic.split("_")
            if all(t in normalized for t in topic_parts):
                self.logger.info(f"Mapped '{normalized.split('-')[0].strip()}' to '{topic}'")
                return topic
        return "entertainment"  # default topic if no match found
    
    def encode_splade_query(self, query):
        device = next(self.splade_model.parameters()).device
        
        inputs = self.splade_tokenizer(
            query, return_tensors="pt", max_length=512, 
            truncation=True, padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = self.splade_model(**inputs)
            logits = outputs.logits
            sparse_weights = torch.log1p(torch.relu(logits))
            doc_rep = torch.max(sparse_weights, dim=1)[0].cpu().squeeze()
            
            # Convert to sparse vector
            non_zero_indices = torch.nonzero(doc_rep > 0.01).squeeze(-1)
            sparse_vector = {
                idx.item(): doc_rep[idx].item() 
                for idx in non_zero_indices
            }
        
        return sparse_vector
    
    def search_splade(self, query_vector, top_k):
        doc_scores = defaultdict(float)
        
        for term_id, query_weight in query_vector.items():
            if str(term_id) in self.inverted_index:
                for doc_idx, doc_weight in self.inverted_index[str(term_id)]:
                    doc_scores[doc_idx] += query_weight * doc_weight
        
        # clean the self.inverted_index to save memory
        self.inverted_index = None
        self.index_data = None
        self.saved_index = None
        torch.cuda.empty_cache()

        # Get top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for rank, (doc_idx, score) in enumerate(sorted_docs):
            results.append({
                'corpus_id': self.corpus_ids[doc_idx],
                'score': score,
                'rank': rank + 1
            })
        
        self.corpus_ids = None
        return results
    
    def load_queries(self):
        queries = []
        with open(self.config["input_queries"], 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                queries.append({
                    'query_id': data['query_id'],
                    'query': data['query']
                })
        return queries
    
    def save_trec_results(self, all_results):
        with open(self.config["output_trec"], 'w') as f:
            for query_id, results in all_results.items():
                for result in results:
                    f.write(f"{query_id} Q0 {result['corpus_id']} {result['rank']} {result['score']:.6f} {run_id}\n")
    
    def run(self):
        # Load queries
        queries = self.load_queries()

        # try first 10 queries
        # queries = queries[:10]
        
        all_results = {}
        
        for query_data in queries:
            query_id = query_data['query_id']
            query_text = query_data['query']
            
            # Classify topic (currently just returns entertainment)
            self.topic_name = self.classify_topic(query_text)
            
            self.load_splade_index()
            self.logger.info(f"Loaded splade index for topic {self.topic_name}")
            # Encode query with SPLADE
            query_vector = self.encode_splade_query(query_text)
            self.logger.info(f"Encoded query with SPLADE")
            # Search
            results = self.search_splade(query_vector, self.config["top_k"])
            self.logger.info(f"Searched for query {query_id} with SPLADE")
            all_results[query_id] = results
        
        # Save TREC results
        self.save_trec_results(all_results)
        self.logger.info(f"Saved TREC results to {self.config['output_trec']}")

if __name__ == "__main__":
    pipeline = SimpleSPLADEPipeline(CONFIG)
    start_time = time.time()
    pipeline.run()
    end_time = time.time()
    print(f"Pipeline completed in {end_time - start_time} seconds")
    print(f"Results saved to {CONFIG['output_trec']}")