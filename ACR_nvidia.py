### Import Libraries
import os
import re
import sys
import json
import warnings
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import acr_prompt, acr_prompt_summary, remove_diffs, myeval, load_checkpoint, save_checkpoint
from nvidia_api import ask_generate, MAX_WORKERS

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results", "results_gpt5.csv")


def load_data(lang=None):
    with open(os.path.join(SCRIPT_DIR, "CodeReviewQA_with_summaries-gpt5.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    if lang:
        data = [ex for ex in data if ex["lang"] == lang]
    return data


def save_csv_row(model, task, value):
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
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
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen2.5-coder-3b-instruct"
    use_summary = "--summary" in sys.argv
    language_type = None
    for arg in sys.argv[2:]:
        if not arg.startswith("--"):
            language_type = arg
            break

    data = load_data(lang=language_type.lower() if language_type else None)

    # Run Inference
    test_prompts = test_prompt(data, language_type or "code", use_summary)

    TASK = "ACR"
    partial = load_checkpoint(TASK, model_name)
    remaining = [r for r in range(len(data)) if r not in partial]

    pbar = tqdm(total=len(data), desc=f"ACR | {model_name}")
    pbar.update(len(partial))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for row in remaining:
            fut = ex.submit(ask_generate, model_name, test_prompts[row], 512, ["[/code]"])
            futures[fut] = row

        saved_at = len(partial)
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                output = fut.result()
            except Exception:
                save_checkpoint(TASK, model_name, partial)
                raise
            gold = "\n".join([line[1:] for line in data[row]["new"].split("\n")])
            eval_df = save_eval(gold, output)
            partial[row] = eval_df.iloc[0].tolist()
            if len(partial) - saved_at >= 25:
                save_checkpoint(TASK, model_name, partial)
                saved_at = len(partial)
            em = sum(rec[1] for rec in partial.values())
            pbar.set_description(f"ACR | {model_name} | em={em}/{len(partial)}")
            pbar.update(1)
    save_checkpoint(TASK, model_name, partial)
    pbar.close()

    rows = [partial[r] for r in range(len(data)) if r in partial]
    c_save = pd.DataFrame(rows, columns=['generation', 'em', 'em_trim', 'em_no_space', 'em_no_comment'])

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
