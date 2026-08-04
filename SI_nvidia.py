### Import Libraries
import os
import sys
import json
import warnings
import itertools
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import si_prompt, si_prompt_summary, ct_formatter, remove_diffs, count_matching_elements, load_checkpoint, save_checkpoint
from nvidia_api import ask_guided, MAX_WORKERS
from ACR_nvidia import save_csv_row, load_data

warnings.filterwarnings("ignore")


### Prompt Constructor (EXACT same as paper)
def prompt_combinations(example, mode, language_type, use_summary):
    symbol_index_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    if mode == "easy":
        options = example["solution_wrong_easy"].copy()
    elif mode == "hard":
        options = example["solution_wrong_hard"].copy()

    options.append(example["solution_correct"])
    all_permutations = list(itertools.permutations(options))

    prompts = []
    correct_symbols = []
    for permutation in all_permutations:
        correct_symbols.append(symbol_index_map[permutation.index(example["solution_correct"])])
        if use_summary:
            prompts.append(si_prompt_summary.format(lang=language_type,
                                                    option_a="\n" + permutation[0],
                                                    option_b="\n" + permutation[1],
                                                    option_c="\n" + permutation[2],
                                                    option_d="\n" + permutation[3],
                                                    code_snippet=remove_diffs(example["old"]),
                                                    code_review=example["review"],
                                                    ct=ct_formatter[example["type_correct"]],
                                                    summary=example.get("summary", "")))
        else:
            prompts.append(si_prompt.format(lang=language_type,
                                            option_a="\n" + permutation[0],
                                            option_b="\n" + permutation[1],
                                            option_c="\n" + permutation[2],
                                            option_d="\n" + permutation[3],
                                            code_snippet=remove_diffs(example["old"]),
                                            code_review=example["review"],
                                            ct=ct_formatter[example["type_correct"]]))

    return prompts, correct_symbols, all_permutations


### Run Test
def main():
    model_name = sys.argv[1]
    mode = sys.argv[2]  # easy or hard
    use_summary = "--summary" in sys.argv
    language_type = None
    for arg in sys.argv[3:]:
        if not arg.startswith("--"):
            language_type = arg
            break

    data = load_data(lang=language_type.lower() if language_type else None)

    TASK = "SIE" if mode == "easy" else "SIH"
    SYMBOLS = ["A", "B", "C", "D"]
    partial = load_checkpoint(TASK, model_name)
    partial = {k: v for k, v in partial.items()
               if isinstance(v, dict) and all(a is not None for a in v.get("answers", []))}
    remaining = [i for i in range(len(data))
                 if i not in partial and data[i]["type_correct"] != "remove_only"]

    pbar = tqdm(total=len(remaining) + len(partial), desc=f"SI-{mode} | {model_name}")
    pbar.update(len(partial))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        pending = {}
        in_flight = {}
        for ex_idx in remaining:
            prompt_permutations, correct_answers, combinations = prompt_combinations(data[ex_idx], mode, language_type or "code", use_summary)
            in_flight[ex_idx] = {
                "combinations": combinations,
                "correct": correct_answers,
                "answers": [None] * len(prompt_permutations),
            }
            for p_idx, prompt in enumerate(prompt_permutations):
                fut = ex.submit(ask_guided, model_name, prompt, SYMBOLS)
                pending[fut] = (ex_idx, p_idx)

        saved_at = len(partial)
        for fut in as_completed(pending):
            ex_idx, p_idx = pending[fut]
            try:
                symbol_probs = fut.result()
            except Exception:
                save_checkpoint(TASK, model_name, partial)
                raise
            in_flight[ex_idx]["answers"][p_idx] = symbol_probs
            if all(a is not None for a in in_flight[ex_idx]["answers"]):
                partial[ex_idx] = in_flight[ex_idx]
                inv = sum(1 for rec in partial.values()
                          if [max(sp, key=sp.get) for sp in rec["answers"]] == rec["correct"])
                pbar.set_description(f"SI-{mode} | {model_name} | inv={inv}/{len(partial)}")
                pbar.update(1)
                if len(partial) - saved_at >= 5:
                    save_checkpoint(TASK, model_name, partial)
                    saved_at = len(partial)
    save_checkpoint(TASK, model_name, partial)
    pbar.close()

    records = []
    for ex_idx in range(len(data)):
        if ex_idx in partial and all(a is not None for a in partial[ex_idx]["answers"]):
            rec = partial[ex_idx]
            records.append([rec["combinations"],
                            rec["answers"],
                            [max(sp, key=sp.get) for sp in rec["answers"]],
                            rec["correct"],
                            data[ex_idx]["type_correct"]])
    c_save = pd.DataFrame(records, columns=['combinations', 'softmax_probs', 'model_answers', 'correct_answers', 'GT'])

    # Output Results
    from utils import calc_results
    calc_results(c_save)

    # Save to CSV
    invariant = len(c_save.loc[c_save['model_answers'] == c_save['correct_answers']])
    total = len(c_save)
    save_csv_row(model_name, TASK, invariant / total * 100)


if __name__ == "__main__":
    main()
