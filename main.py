import os
import sys
import gc
import subprocess 
import pandas as pd
from ACR_ollama import save_csv_row 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

RESULTS_CSV = "results/results.csv"
MODELS = ["llama3.2:latest","qwen2.5-coder:3b"]
DIR = os.path.dirname(os.path.abspath(__file__))


def already_done(model, task):
    try:
        df = pd.read_csv(RESULTS_CSV)
        if model in df["Model"].values:
            val = df.loc[df["Model"] == model, task].values[0]
            return pd.notna(val)
        return False
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False


def run(model_name, task_name, cmd):
    if already_done(model_name, task_name):
        print(f"SKIP {task_name} | {model_name} (already done)")
        return
    print(f"\n{'='*60}")
    print(f"  {task_name} | {model_name}")
    print(f"{'='*60}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    subprocess.run(cmd, shell=True, env=env)
    gc.collect()


def main():
    for model in MODELS:
        run(model, "ACR",  f"python ACR_ollama.py {model} --summary")
        run(model, "CTR",  f"python CTR_ollama.py {model} --summary")
        run(model, "CLE",  f"python CL_ollama.py {model} easy --summary")
        run(model, "CLH",  f"python CL_ollama.py {model} hard --summary")
        run(model, "SIE",  f"python SI_ollama.py {model} easy --summary")
        run(model, "SIH",  f"python SI_ollama.py {model} hard --summary")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    try:
        df = pd.read_csv(RESULTS_CSV)
        if df.empty:
            print("No results yet.")
        else:
            print(df.to_string(index=False))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("No results yet.")


if __name__ == "__main__":
    main()
