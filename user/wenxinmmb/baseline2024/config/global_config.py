PROCESS_SUBSET = True
SUBSET_SIZE = 1000000 # 1 million. The number of documents to process. Use in local development to avoid memory issues. Effective when PROCESS_SUBSET is True.
METRICS = "recall_10,recall_1000,ndcg_cut_10,ndcg_cut_1000,recip_rank"
