import os
import json
import pandas as pd

splits = ['train', 'dev1', 'dev2', 'dev3', 'llmset1-train', 'llmset1-dev']
methods = ['bm25', 'bge']
versions = ['v4', 'v5', 'v6']
cutoffs = ['10', '100', '1000']

base_dir = 'outputs/scores'

def read_stats(path):
    with open(path, 'r') as f:
        return json.load(f)

for split in splits:
    for method in methods:
        results = {'Cutoff': cutoffs}
        baseline = None
        for version in versions:
            stats_path = os.path.join(base_dir, f"{split}-{method}", version, 'rerank-stats.json')
            if not os.path.exists(stats_path):
                print(f"Missing: {stats_path}")
                continue
            stats = read_stats(stats_path)
            if baseline is None:
                baseline = stats['Baseline Results']
                results['Baseline NDCG'] = [baseline['NDCG'][c] for c in cutoffs]
                results['Baseline Recall'] = [baseline['Recall'][c] for c in cutoffs]
            results[f'{version} NDCG'] = [
                f"{stats['Reranked Results']['NDCG'][c]:.4f} ({stats['Changes']['NDCG'][c]['Percentage Change']:.2f}%)"
                for c in cutoffs
            ]
            results[f'{version} Recall'] = [
                f"{stats['Reranked Results']['Recall'][c]:.4f} ({stats['Changes']['Recall'][c]['Percentage Change']:.2f}%)"
                for c in cutoffs
            ]
        df = pd.DataFrame(results)
        columns_order = [
            'Cutoff',
            'Baseline NDCG', 'v4 NDCG', 'v5 NDCG', 'v6 NDCG',
            'Baseline Recall', 'v4 Recall', 'v5 Recall', 'v6 Recall'
        ]
        df = df[columns_order]
        print(f"\n=== {split} | {method} ===")
        print(df.to_markdown(index=False))