# LLM reranking module

## Ollama setup
1. Install Ollama
```
$ curl -fsSL https://ollama.com/install.sh | sh
```
Verify ollama version
```
$ ollama --version
```

2. Download Gemma 3 12B Model using pull command
```
$ ollama pull gemma3:12b
```

3. Run the model
```
$ ollama run gemma3:12b
```

4. Run Ollama as a server
```
$ ollama serve
```

5. Ollama runs on port 11434 by default. Check Ollama is running on port 11434.
```
$ curl http://localhost:11434/api/tags
```
You should see output like

```
{"models":[{"name":"gemma3:12b","model":"gemma3:12b","modified_at":"2025-07-17T23:07:13.757923686-07:00","size":8149190253,"digest":"f4031aab637d1ffa37b42570452ae0e4fad0314754d17ded67322e4b95836f8a","details":{"parent_model":"","format":"gguf","family":"gemma3","families":["gemma3"],"parameter_size":"12.2B","quantization_level":"Q4_K_M"}}]}
```

### Run script to verify ollama setup 
The script makes a request to rerank three documents regarding one query. By default the request is directed to the port number 11434 and the model used is "gemma3:12b". 
```
$ python test_ollama.py
```

You can specify the port number and model arguments if they are different from the defaults.
```
$ python test_ollama.py --port PORT_NUMBER --model MODEL_NAME
``` 

You can expect to see the following output
```
Testing Ollama reranking on port 11434
Using model: gemma3:12b
Query: What is the capital of France?

Original documents:
1. London is the capital city of England and the United Kingdom. It is situated on the River Thames in southeast England.
2. Paris is the capital and most populous city of France. It is located in the north-central part of the country.
3. Berlin is the capital and largest city of Germany. It is located in northeastern Germany on the banks of the rivers Spree and Havel.

==================================================
Making request to Ollama server...
==================================================

Raw Ollama Response:
{"ranking": [
    {"index": 1, "score": 0.98},
    {"index": 0, "score": 0.05},
    {"index": 2, "score": 0.02}
]}
```
## Install rank-llm python package
1. Create a virtual enviroment (optional)
```
$ pyenv virtualenv 3.10.12 rank-llm-env
$ pyenv activate rank-llm-env
```

2. download the rank-llm library repository. Here you need to download the forked repository because it has openai compatible backend.
```
make sure you are in the current directory
$ pwd
$ git clone https://github.com/wenxinmmb/rank_llm.git
$ cd rank_llm/
$ pip install -e . # do this so that "rank_llm" library points to local forked directory
```
3. Check the dependency is installed correctly
```
$ python -c "import rank_llm
> print(rank_llm.__file__)"

(output)
/home/wenxin/project/trec-tot-2025/user/wenxinmmb/rerank_module/rank_llm/src/rank_llm/__init__.py
```

## Run the rerank script
```
$ python tot_llm_reranking.py
