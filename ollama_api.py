import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


def ask_generate(model, prompt, max_tokens=512, stop=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens}
    }
    if stop:
        body["options"]["stop"] = stop
    response = requests.post(OLLAMA_URL, json=body)
    data = response.json()
    return type("Output", (), {"text": data["message"]["content"]})()


def ask_guided(model, prompt, choices):
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
        "format": {
            "type": "object",
            "properties": {"answer": {"enum": choices}},
            "required": ["answer"]
        },
        "logprobs": True
    })
    data = response.json()
    parsed = json.loads(data["message"]["content"])

    logprobs_list = data.get("logprobs", [])
    symbol_probs = {}
    for c in choices:
        symbol_probs[c] = -9999
    for entry in logprobs_list:
        token = entry["token"].strip()
        if token in symbol_probs:
            symbol_probs[token] = entry["logprob"]

    return symbol_probs
