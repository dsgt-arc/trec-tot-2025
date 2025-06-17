# 1. Similarity search with precomputed BGE-M3 embeddings
1. Download the embedding parquet files -- Upstash/wikipedia-2024-06-bge-m3, by running the following script
```
$ python hf_download.py --local_dir (your_local_directory_to_store_data)
```

2. Run the dense_search_bge.py script to generate search results. This will output top_k for every partition.
```
$ python dense_search_bge.py --data_version 2025 --data_path $DATA_PATH
```

3. Run aggregation.py script to aggregate the results and generate run file.
```
$ python aggregation.py
```

# 2. Dense Retrieval Baseline

This readme contains instructions for reproducing a Dense Retrieval baseline run using [Sentence Transformers](https://www.sbert.net/)
for the TREC track on tip-of-the-tongue (ToT)  retrieval. 

## Embedding and perfoming dense retrieval using distilBERT
Command to embed the 2025 dataset with baseline distilbert model,
and run dense retrieval.
```
python train_dense.py \
--model_or_checkpoint distilbert-base-uncased \
--model_dir dense_models/2025/distilbert_inf/ \
--embedding_dir dense_embeddings/2025/distilbert_inf \
--negatives_out distilbert_negatives/2025/distilbert_inf \
--data_path $DATA_PATH \
--loss_fn triplet \
--loss_distance cosine \
--encode_norm \
--run_id distilbert_inf \
--epochs 0 \
--embed_size 768 \
--encode_after_train \
--no_train
```
------
## The information below are out of date. They are from the original repository. It may not be compatible with the current code. Kept for reference only.

## Finetune distil bert with hard negatives
Note that the script below trains on *both* the dev1 and train splits. 
```
# step 1: train a dense retrieval model (baseline-distilbert) and generate negatives
python train_dense.py \
--model_dir dense_models/baseline_distilbert_0/ \
--embedding_dir dense_embeddings/2025/run_1 \
--data_path $DATA_PATH \
--encode_after_train \
--epochs 20 --loss_margin 0.75 --lr 6e-05 --n_train_negatives 5  \
--run_id baseline_distilbert_0 --device cuda --weight_decay 0.01 \
--model_or_checkpoint distilbert-base-uncased --embed_size 768 \
--encode_batch_size 128 --batch_size 10 --loss_fn triplet \
--loss_distance cosine --encode_norm \
--negatives_out distilbert_negatives  >> distilbert.log 2>&1 &

# step 2: load the new negatives and retrain
python train_dense.py \
--model_dir dense_models/baseline_distilbert/ \
--data_path $DATA_PATH \
--negatives_path distilbert_negatives \
--epochs 20 --loss_margin 0.75 --lr 6e-05 --n_train_negatives 5  \
--run_id baseline_distilbert --device cuda --weight_decay 0.01 \
--model_or_checkpoint distilbert-base-uncased --embed_size 768 \
--encode_batch_size 128 --batch_size 10 --loss_fn triplet \
--loss_distance cosine --encode_norm  >> distilbert.log 2>&1 &
```


Inference: 
(TODO: need to update the command)
```
srun -p gpu --time=36:00:00 --mem=62G --gres=gpu:nvidia_rtx_a6000:2 python train_dense.py \
    --model_or_checkpoint dense_models/baseline_distilbert/model \
    --model_dir dense_models/baseline_distilbert_out/ \
    --device cuda \
    --loss_fn triplet \
    --loss_distance cosine \
    --encode_norm \
    --run_id baseline_distilbert \
    --no_train \
    --epochs 0 \
    --embed_size 768 \
    --data_path $DATA_PATH \
    --encode_after_train \
    --encode_batch_size 386

```

You can evaluate either using pytrec_eval (included in the script above), or use `trec_eval`:
```
trec_eval -m ndcg_cut.10,1000 -m recall.1000  -m recip_rank $DATA_PATH/dev1-2025/qrel.txt ./dense_models/baseline_distilbert/dev1.run
recip_rank            	all	0.0901
recall_1000           	all	0.5600
ndcg_cut_10           	all	0.1040
ndcg_cut_1000         	all	0.1665

```