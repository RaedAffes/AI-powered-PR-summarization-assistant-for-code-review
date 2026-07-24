import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))


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


def _single_guided(model, prompt, choices):
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 32},
        "format": {
            "type": "object",
            "properties": {"answer": {"enum": choices}},
            "required": ["answer"]
        },
        "logprobs": True
    })
    data = response.json()
    content = data["message"]["content"].strip()
    if not content:
        symbol_probs = {c: -9999 for c in choices}
        symbol_probs[choices[0]] = 0
        return symbol_probs
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        symbol_probs = {c: -9999 for c in choices}
        symbol_probs[choices[0]] = 0
        return symbol_probs

    logprobs_list = data.get("logprobs", [])
    symbol_probs = {}
    for c in choices:
        symbol_probs[c] = -9999
    for entry in logprobs_list:
        token = entry["token"].strip()
        if token in symbol_probs:
            symbol_probs[token] = entry["logprob"]

    return symbol_probs


def ask_guided(model, prompt, choices):
    return _single_guided(model, prompt, choices)


def ask_guided_batch(model, prompts, choices, max_workers=None):
    if max_workers is None:
        max_workers = MAX_WORKERS
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_single_guided, model, prompt, choices): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results
