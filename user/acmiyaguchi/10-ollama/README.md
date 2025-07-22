# ollama

This is to deal with experiments with ollama on PACE.
We need an openai-compatible API for reranking (and other basic tasks), so getting this figured out is important for testing out various methods.

## notes

### running ollama

Luckily, ollama is available on PACE:

```bash
module spider ollama
...
     Versions:
        ollama/0.5.1
        ollama/0.6.6
        ollama/0.9.0
```

Now we can load and test that it's running.
Start up an interactive session (e.g. `salloc-gpu`) and then start a terminal multiplexer like `tmux`.
Ensure that `OLLAMA_MODELS` is set and is pointing to the `scratch` directory.
The best way to do this is to add a new line to the bash startup script (e.g. `~/.bashrc`):

```bash
# in ~/.bashrc or similar
export OLLAMA_MODELS=$HOME/scratch/ollama_models
```

Then we can load the module and start the server:

```bash
# in the serving terminal window
module load ollama
ollama serve

# in a new terminal window
ollama run gemma3:12b
```

This should get you into a terminal where you can type messages to the model.
We can check that the server is running via curl.

```
$ curl http://localhost:11434/api/tags | jq
{
  "models": [
    {
      "name": "gemma3:12b",
      "model": "gemma3:12b",
      "modified_at": "2025-07-22T03:11:37-04:00",
      "size": 8149190253,
      "digest": "f4031aab637d1ffa37b42570452ae0e4fad0314754d17ded67322e4b95836f8a",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "gemma3",
        "families": [
          "gemma3"
        ],
        "parameter_size": "12.2B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

We can verify that the model files are in the `OLLAMA_MODELS` directory:

```bash
$ tree ~/scratch/ollama_models/
/storage/home/hcoda1/8/amiyaguchi3/scratch/ollama_models/
├── blobs
│   ├── sha256-3116c52250752e00dd06b16382e952bd33c34fd79fc4fe3a5d2c77cf7de1b14b
│   ├── sha256-6819964c2bcf53f6dd3593f9571e91cbf2bab9665493f870f96eeb29873049b4
│   ├── sha256-dd084c7d92a3c1c14cc09ae77153b903fd2024b64a100a0cc8ec9316063d2dbc
│   ├── sha256-e0a42594d802e5d31cdc786deb4823edb8adff66094d49de8fffe976d753e348
│   └── sha256-e8ad13eff07a78d89926e9e8b882317d082ef5bf9768ad7b50fcdbbcd63748de
└── manifests
    └── registry.ollama.ai
        └── library
            └── gemma3
                └── 12b
```

### benchmarking ollama

We can benchmark ollama using a pre-built script: https://github.com/larryhopecode/ollama-benchmark.
This one seemed straightforward to use:

```bash
$ nvidia-smi
Tue Jul 22 03:27:53 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro RTX 6000                On  |   00000000:AF:00.0 Off |                  Off |
| 33%   39C    P8             16W /  260W |   10406MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2118095      C   ...kages/ollama/0.9.0/bin/ollama      10402MiB |
+-----------------------------------------------------------------------------------------+

$ python ollama-benchmark/benchmark.py --verbose --models gemma3:12b --prompts "what is tip of the tongue?"

----------------------------------------------------
        Model: gemma3:12b
        Performance Metrics:
            Prompt Processing:  860.92 tokens/sec
            Generation Speed:   48.75 tokens/sec
            Combined Speed:     49.79 tokens/sec

        Workload Stats:
            Input Tokens:       16
            Generated Tokens:   705
            Model Load Time:    0.06s
            Processing Time:    0.02s
            Generation Time:    14.46s
            Total Time:         14.54s
----------------------------------------------------
```
