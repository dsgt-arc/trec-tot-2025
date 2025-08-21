# Run 1
## Model
gemini-2.5-flash
Parameter:
max-token:5000
temperature: 0.0
prompt: 
"Identify up to 20 entities that are titles of Wikipedia pages which answer the following tip-of-the-tongue query. "
    "For each entity, provide the Wikipedia page title and a relevance score from 1-5, where:\n"
    "- 1 = Irrelevant to the query\n"
    "- 2 = Somewhat relevant\n"
    "- 3 = Moderately relevant\n"
    "- 4 = Highly relevant\n"
    "- 5 = Most relevant and directly answers the query\n\n"
    "Return a JSON object containing the entity titles and their relevance scores.\n\n"
    "TOT Query: {query}"

## Search parameters
MIN_SCORE = 100
BM25_K = 10

## Dataset
dev3 first 100

## Performance
~/project-v2/trec-tot-2025/user/wenxinmmb/baseline2024/llm$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/dev3-2025/qrel-first-100.txt outputs/scored/runs/dev3-gemini-100-10-v6.txt
recip_rank              all     0.6475
recall_1000             all     0.7000
ndcg_cut_10             all     0.6609
ndcg_cut_1000           all     0.6609

# Run 2

## Model
gemini-2.5-flash
Parameter:
max-token:5000
temperature: 0.0
prompt: 
"Identify the single best entity that is a title of a Wikipedia page which answers the following tip-of-the-tongue query. "
    "Provide the Wikipedia page title and a relevance score from 1-5, where:\n"
    "- 1 = Irrelevant to the query\n"
    "- 2 = Somewhat relevant\n"
    "- 3 = Moderately relevant\n"
    "- 4 = Highly relevant\n"
    "- 5 = Most relevant and directly answers the query\n\n"
    "Return a JSON object containing the best matching entity title and its relevance score.\n\n"
    "TOT Query: {query}"

## Search parameters
MIN_SCORE = 100
BM25_K = 10

## Matching counters
Counter({'exact_1': 95, 'strategy3_1': 1, 'strategy2_1': 1, 'exact_n': 1})
unmatched: 2

## Dataset
dev3 first 100

## Performance
wenxin@WX-PC:~/project-v2/trec-tot-2025/user/wenxinmmb/baseline2024/llm$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/dev3-2025/qrel-first-100.txt outputs/scored/runs/dev3-gemini-100-10-v7.txt
recip_rank              all     0.6500
recall_1000             all     0.6500
ndcg_cut_10             all     0.6500
ndcg_cut_1000           all     0.6500

# Run 3

## Model
gpt-4o-mini
Parameter:
max-token:5000
temperature: 0.0
prompt: 
"Identify the single best entity that is a title of a Wikipedia page which answers the following tip-of-the-tongue query. "
    "Provide the Wikipedia page title and a relevance score from 1-5, where:\n"
    "- 1 = Irrelevant to the query\n"
    "- 2 = Somewhat relevant\n"
    "- 3 = Moderately relevant\n"
    "- 4 = Highly relevant\n"
    "- 5 = Most relevant and directly answers the query\n\n"
    "Return a JSON object containing the best matching entity title and its relevance score.\n\n"
    "TOT Query: {query}"

## Search parameters
MIN_SCORE = 100
BM25_K = 10

## Matching counters
Counter({'exact_1': 94, 'strategy3_1': 1})
unmatched: 2

## Dataset
dev3 first 100

## Performance
(trec-tot-2025) wenxin@WX-PC:~/project-v2/trec-tot-2025/user/wenxinmmb/baseline2024/llm$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/dev3-2025/qrel-first-100.txt outputs/scored/runs/dev3-gemini-100-10-v8.txt
recip_rank              all     0.6900
recall_1000             all     0.6900
ndcg_cut_10             all     0.6900
ndcg_cut_1000           all     0.6900
