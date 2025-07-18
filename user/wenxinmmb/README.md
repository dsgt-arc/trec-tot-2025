# my-username

This is a scratch directory where you can commit files to without worrying about polluting the main repository.
Use it to test out new ideas or to fork code/notebooks from other people and to share the results with them.

# Ollama hosting steps
1. curl -fsSL https://ollama.com/install.sh | sh
2. ollama --version
3. ollama pull gemma3:12b # download gemma model
4. ollama run gemma3:12b # run the model, start chatting
5. ollama serve # start server

## Ollama interfacing
$ pip install ollama
```
from ollama import chat

response = chat(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}]
)

print(response["message"]["content"])
```

## The openai version
```
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',  # Ollama's local API
    api_key='ollama'  # Required but not validated
)

response = client.chat.completions.create(
    model='gemma3:12b',
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```