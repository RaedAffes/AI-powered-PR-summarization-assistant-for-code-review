import json
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)


def ask_generate(model, prompt, max_tokens=512, stop=None):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    response = client.chat.completions.create(**kwargs)
    return type("Output", (), {"text": response.choices[0].message.content})()


def ask_guided(model, prompt, choices):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True,
        top_logprobs=10,
    )

    content = response.choices[0].message.content.strip()
    if not content:
        symbol_probs = {c: -9999 for c in choices}
        symbol_probs[choices[0]] = 0
        return symbol_probs

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None

    symbol_probs = {c: -9999 for c in choices}

    if parsed and isinstance(parsed, dict) and "answer" in parsed:
        answer = parsed["answer"]
        if answer in symbol_probs:
            symbol_probs[answer] = 0
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            for entry in response.choices[0].logprobs.content:
                token = entry.token.strip()
                if token in symbol_probs:
                    symbol_probs[token] = entry.logprob
        if all(v == -9999 for v in symbol_probs.values()):
            symbol_probs[choices[0]] = 0
        return symbol_probs

    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        for entry in response.choices[0].logprobs.content:
            token = entry.token.strip()
            if token in symbol_probs:
                symbol_probs[token] = entry.logprob

    if all(v == -9999 for v in symbol_probs.values()):
        for c in choices:
            if content.strip().upper() == c.upper():
                symbol_probs[c] = 0
                break
        else:
            symbol_probs[choices[0]] = 0

    return symbol_probs
