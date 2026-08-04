import os
import sys
import gc
import subprocess
import pandas as pd
from ACR_nvidia import save_csv_row

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(DIR, "results", "results_gpt5.csv")
MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    #"meta-llama/llama-3.3-70b-instruct",
    "microsoft/phi-4",
    "google/gemma-2-27b-it"
    ]    


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
        run(model, "ACR",  f"python {os.path.join(DIR, 'ACR_nvidia.py')} {model} --summary")
        run(model, "CTR",  f"python {os.path.join(DIR, 'CTR_nvidia.py')} {model} --summary")
        run(model, "CLE",  f"python {os.path.join(DIR, 'CL_nvidia.py')} {model} easy --summary")
        run(model, "CLH",  f"python {os.path.join(DIR, 'CL_nvidia.py')} {model} hard --summary")
        run(model, "SIE",  f"python {os.path.join(DIR, 'SI_nvidia.py')} {model} easy --summary")
        run(model, "SIH",  f"python {os.path.join(DIR, 'SI_nvidia.py')} {model} hard --summary")

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
