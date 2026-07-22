### Import Libraries
import os
import sys
import json
import warnings
import itertools
import pandas as pd
from tqdm import tqdm
from utils import si_prompt, si_prompt_summary, ct_formatter, remove_diffs, count_matching_elements
from ollama_api import ask_guided
from ACR_ollama import save_csv_row, load_data

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


### Evaluation (EXACT same logic as paper)
def test_example(example, model_name, mode, language_type, use_summary):
    prompt_permutations, correct_answers, combinations = prompt_combinations(example, mode, language_type, use_summary)
    model_answers = []

    for prompt in prompt_permutations:
        symbol_probs = ask_guided(model_name, prompt, ["A", "B", "C", "D"])
        model_answers.append(symbol_probs)

    example_record = [combinations,
                      model_answers,
                      [max(symbol_probs, key=symbol_probs.get) for symbol_probs in model_answers],
                      correct_answers,
                      example["type_correct"]]

    return pd.DataFrame([example_record], columns=['combinations', 'softmax_probs', 'model_answers', 'correct_answers', 'GT'])


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

    c_save = pd.DataFrame(columns=['combinations', 'softmax_probs', 'model_answers', 'correct_answers', 'GT'])
    pbar = tqdm(range(len(data)), desc=f"SI-{mode} | {model_name}")
    for row in pbar:
        if data[row]["type_correct"] != "remove_only":
            example_save = test_example(data[row], model_name, mode, language_type or "code", use_summary)
            c_save = pd.concat([c_save, example_save])
            inv = len(c_save.loc[c_save['model_answers'] == c_save['correct_answers']])
            pbar.set_description(f"SI-{mode} | {model_name} | inv={inv}/{len(c_save)}")

    # Output Results
    from utils import calc_results
    calc_results(c_save)

    # Save to CSV
    invariant = len(c_save.loc[c_save['model_answers'] == c_save['correct_answers']])
    total = len(c_save)
    task_name = "SIE" if mode == "easy" else "SIH"
    save_csv_row(model_name, task_name, invariant / total * 100)


if __name__ == "__main__":
    main()
