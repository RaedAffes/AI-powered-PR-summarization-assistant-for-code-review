### Import Libraries
import os
import re
import sys
import warnings
import pandas as pd
from tqdm import tqdm
from utils import acr_prompt, acr_prompt_summary, remove_diffs, myeval
from ollama_api import ask_generate

warnings.filterwarnings("ignore")

RESULTS_CSV = "results/results.csv"


def load_data(lang=None):
    with open("AI-powered-PR-summarization-assistant-for-code-review\\CodeReviewQA_with_summaries.json", "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    if lang:
        data = [ex for ex in data if ex["lang"] == lang]
    return data


import json


def save_csv_row(model, task, value):
    os.makedirs("results", exist_ok=True)
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
    else:
        df = pd.DataFrame(columns=["Model", "ACR", "CTR", "CLE", "CLH", "SIE", "SIH"])
    if model not in df["Model"].values:
        df = pd.concat([df, pd.DataFrame([{"Model": model}])], ignore_index=True)
    df.loc[df["Model"] == model, task] = round(value, 2)
    df.to_csv(RESULTS_CSV, index=False)


### Prompt Constructor
def test_prompt(test_set, language_type, use_summary):
    test_prompts = []
    for row in tqdm(range(len(test_set)), desc="Building prompts"):
        example = test_set[row]
        code_snippet = remove_diffs(example["old"])
        if use_summary:
            prompt = acr_prompt_summary.format(lang=language_type,
                                               code_snippet=code_snippet,
                                               code_review=example["review"],
                                               summary=example.get("summary", ""))
        else:
            prompt = acr_prompt.format(lang=language_type,
                                       code_snippet=code_snippet,
                                       code_review=example["review"])
        test_prompts.append(prompt)
    return test_prompts


### Evaluation (EXACT same as paper)
def save_eval(gold, output):
    raw = output.text.strip()
    raw = re.sub(r'^```[^\n]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    raw = re.sub(r'^\[[^\]]+\]\n?', '', raw)
    raw = re.sub(r'\n?\[/[^\]]+\]$', '', raw)
    generated = "\n".join([line[2:] for line in raw.split("\n")])
    result = myeval(gold, generated)
    record = [generated] + list(result)
    return pd.DataFrame([record], columns=['generation', 'em', 'em_trim', 'em_no_space', 'em_no_comment'])


### Run Test
def main():
    model_name = "qwen2.5-coder:3b"  # e.g. llama3.2:latest
    use_summary = "--summary" in sys.argv
    language_type = None
    for arg in sys.argv[2:]:
        if not arg.startswith("--"):
            language_type = arg
            break

    data = load_data(lang=language_type.lower() if language_type else None)

    # Run Inference
    test_prompts = test_prompt(data, language_type or "code", use_summary)

    c_save = pd.DataFrame(columns=['generation', 'em', 'em_trim', 'em_no_space', 'em_no_comment'])
    pbar = tqdm(range(len(data)), desc=f"ACR | {model_name}")
    for row in pbar:
        output = ask_generate(model_name, test_prompts[row], max_tokens=512, stop=["[/code]"])
        gold = "\n".join([line[1:] for line in data[row]["new"].split("\n")])
        eval_df = save_eval(gold, output)
        c_save = pd.concat([c_save, eval_df])
        pbar.set_description(f"ACR | {model_name} | em={int(c_save.em.sum())}/{row+1}")

    # Output Results
    total = len(c_save)
    em = c_save.em.sum()
    em_tr = c_save.em_trim.sum()
    em_ns = c_save.em_no_space.sum()
    em_nc = c_save.em_no_comment.sum()
    print(f"\nACR Results ({total} examples):")
    print(f"  EM:          {em}/{total} = {em/total*100:.1f}%")
    print(f"  EM_TRIM:     {em_tr}/{total} = {em_tr/total*100:.1f}%")
    print(f"  EM_NO_SPACE: {em_ns}/{total} = {em_ns/total*100:.1f}%")
    print(f"  EM_NO_COMMENT: {em_nc}/{total} = {em_nc/total*100:.1f}%")

    # Save to CSV
    save_csv_row(model_name, "ACR", em / total * 100)


if __name__ == "__main__":
    main()
