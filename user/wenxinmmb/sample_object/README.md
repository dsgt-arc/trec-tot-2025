# Wikipedia LLM Query Generation Pipeline

This directory contains tools for creating LLM generated queries from Wikipedia articles through a multi-step pipeline: analyzing Wikipedia articles, storing data in SQLite, sampling entities, generating queries, and preparing train/dev/test datasets.

## Pipeline Overview

### 1. Wikipedia Article Analysis
**Analyze word counts and pageviews of English Wikipedia articles**

**Files:**
- `word_count_analysis.py` - Analyzes word count distribution from Wikipedia corpus
- `wikipedia_pageview_aggregator.py` - Aggregates pageview data from Wikimedia dumps
- `test_read_pgfile.py` - Test file for pageview aggregator

**Commands:**
```bash
# Analyze word counts from corpus
python word_count_analysis.py

# Aggregate pageviews (2022-11-01 to 2023-10-31)  
python wikipedia_pageview_aggregator.py
```

### 2. Database Setup
**Create SQLite database to store Wikipedia page metadata**

**Files:**
- `setup_database.py` - Creates SQLite database schema
- `wikipedia_db.py` - Database connection and operations
- `import_data.py` - Imports data into database

**Commands:**
```bash
# Create database schema
python setup_database.py

# Import Wikipedia data
python import_data.py
```

### 3. Sampling
**Sample 5k entities from the SQLite database based on word count, infobox template information and pageview criteria**

**Files:**
- `run_sampling.py` - Main sampling script with percentile-based filtering
- `test_sampling.py` - Test script for sampling functionality

**Commands:**
```bash
# Run full sampling (5k articles)
python run_sampling.py

# Test sampling with limited categories  
python run_sampling.py --test-mode
```

### 4. LLM Query Generation
**Generate forum-style queries from sampled Wikipedia articles**

**Files:**
- `self_llm_query_gen.py` - Main query generation script
- `templates.py` - Query templates and prompts
- `generate_trec_query.py` - TREC-format query generation

**Commands:**
```bash
# Generate queries for articles 10-100
python self_llm_query_gen.py --start 10 --end 100 --paragraphs-file outputs/generated_query/openai_gpt-4o-mini/json_files/

# Generate TREC format queries
python generate_trec_query.py
```

### 5. QREL and Query File Generation
**Convert generated queries to TREC evaluation format**

**Files:**
- `convert_to_qrel.py` - Converts generated data to QREL format

**Commands:**
```bash
# Convert to QREL format
python convert_to_qrel.py
```

### 6. Dataset Splitting
**Split generated queries into train/dev/test sets**

**Files:**
- `split_train_test.py` - Splits 5k queries into train/dev/test datasets

**Commands:**
```bash
# Split dataset into train/dev/test
python split_train_test.py
```

## Output Structure

```
outputs/
├── word_count_analysis.csv          # Word count statistics
├── page_view/                       # Pageview aggregation results  
├── wikipedia_data.db               # SQLite database
├── sampled_wikipedia_articles.csv # Sampled 5k articles
├── generated_query/                # LLM generated queries
└── train/dev/test/                 # Split datasets
```

## Quick Start

```bash
# 1. Setup database
python setup_database.py

# 2. Sample articles  
python run_sampling.py

# 3. Generate queries
python self_llm_query_gen.py --start 0 --end 5000

# 4. Convert to QREL format
python convert_to_qrel.py

# 5. Split datasets
python split_train_test.py
```