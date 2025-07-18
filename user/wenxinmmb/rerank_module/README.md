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

## Setup rank-llm python package
1. clone github repo
2. pip install -e .

## Test setup

## Run rerank script
python 