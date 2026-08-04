import json
import os
import time
import httpx
from openai import OpenAI, RateLimitError, InternalServerError, APIConnectionError, APITimeoutError

MAX_WORKERS = 12

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    timeout=httpx.Timeout(120.0, connect=10.0),
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "CodeReviewQA Evaluation",
    },
)


def _create(**kwargs):
    delay = 2.0
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            retry_after = None
            try:
                retry_after = e.response.headers.get("retry-after")
            except Exception:
                pass
            wait = delay
            if retry_after:
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = delay
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
        except (InternalServerError, APIConnectionError, APITimeoutError):
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("Exhausted retries")


def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content else ""


def _logprobs_of(response, choices):
    symbol_probs = {c: -9999 for c in choices}
    if response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content:
        for entry in response.choices[0].logprobs.content:
            token = getattr(entry, "token", "") or ""
            logprob = getattr(entry, "logprob", -9999)
            t = token.strip()
            if t in symbol_probs:
                symbol_probs[t] = logprob
    return symbol_probs


def ask_generate(model, prompt, max_tokens=512, stop=None):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    response = _create(**kwargs)
    return type("Output", (), {"text": _extract_text(response.choices[0].message.content)})()


def ask_guided(model, prompt, choices):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50,
        logprobs=True,
        top_logprobs=10,
    )
    try:
        response = _create(**kwargs)
    except Exception:
        kwargs.pop("logprobs", None)
        kwargs.pop("top_logprobs", None)
        response = _create(**kwargs)

    content = _extract_text(response.choices[0].message.content).strip()
    symbol_probs = _logprobs_of(response, choices)

    if all(v == -9999 for v in symbol_probs.values()):
        if not content:
            symbol_probs[choices[0]] = 0
        else:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "answer" in parsed:
                    content = parsed["answer"]
            except json.JSONDecodeError:
                pass
            for c in choices:
                if content.strip().upper() == c.upper():
                    symbol_probs[c] = 0
                    break
            else:
                symbol_probs[choices[0]] = 0

    return symbol_probs
