# wenxinmmb


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