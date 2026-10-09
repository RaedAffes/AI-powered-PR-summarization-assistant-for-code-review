# AI-Powered PR Summarization Assistant for Code Review

Does giving a language model a generated pull request summary help it understand code reviews better?

This project extends the [CodeReviewQA](https://github.com/hongyi-tom/CodeReviewQA) benchmark (Lin et al., ACL 2025) by injecting an AI-generated PR summary into the review context, then measuring the effect on six code review comprehension tasks.

## How It Works

1. **Generate summaries** — `GenerateSummary.py` writes a PR summary for every benchmark example using Llama 3.2 through a local [Ollama](https://ollama.com/) instance, producing `CodeReviewQA_with_summaries.json`.
2. **Run evaluations** — each task script (`ACR_ollama.py`, `CTR_ollama.py`, `CL_ollama.py`, `SI_ollama.py`) runs a model with and without the summary and scores its output against the gold answer.
3. **Aggregate** — `main.py` loops over both models and all six tasks, skipping anything already present in `results/results.csv`.

## Models

Both models are small enough to run locally and are served through Ollama via the thin HTTP client in `ollama_api.py`.

| Model | Ollama tag |
|-------|------------|
| Qwen2.5-Coder-3B-Instruct | `qwen2.5-coder:3b` |
| Llama-3.2-3B-Instruct | `llama3.2:latest` |

## Tasks

| Abbreviation | Task | Type | Metric |
|--------------|------|------|--------|
| ACR | Automated Code Refinement | Generative | Exact Match % |
| CTR | Change Type Recognition | Multiple choice | Accuracy % |
| CLE | Change Localisation (Easy) | Multiple choice | Accuracy % |
| CLH | Change Localisation (Hard) | Multiple choice | Accuracy % |
| SIE | Solution Identification (Easy) | Multiple choice | Accuracy % |
| SIH | Solution Identification (Hard) | Multiple choice | Accuracy % |

## Setup

```bash
pip install -r requirements.txt
pip install datasets openai   # only needed to regenerate summaries
```

Install Ollama and pull the models:

```bash
ollama pull llama3.2
ollama pull qwen2.5-coder:3b
```

## Usage

```bash
# 1. Generate the summary-augmented dataset (once)
python GenerateSummary.py

# 2. Run every task for both models, with summaries
python main.py
```

Or run a single task:

```bash
python ACR_ollama.py llama3.2:latest --summary
python CTR_ollama.py qwen2.5-coder:3b --summary
python CL_ollama.py llama3.2:latest easy --summary
python SI_ollama.py qwen2.5-coder:3b hard --summary
```

Drop the `--summary` flag to evaluate without the injected summary (baseline). Results are appended to `results/results.csv`; re-running `main.py` resumes from where it left off.

## Results

Baseline scores from the original CodeReviewQA setup (code snippet + review comment only):

| Model | ACR | CTR | CLE | CLH | SIE | SIH |
|-------|-----|-----|-----|-----|-----|-----|
| Qwen2.5-Coder-3B-Instruct | 30.3 | 77.7 | 1.8 | 1.8 | 12.2 | 8.0 |
| Llama-3.2-3B-Instruct | 25.9 | 78.8 | 0.8 | 0.4 | 9.9 | 7.6 |

Scores with the AI-generated summary (run in progress):

| Model | ACR | CTR | CLE | CLH | SIE | SIH |
|-------|-----|-----|-----|-----|-----|-----|
| Qwen2.5-Coder-3B-Instruct | 28.7 | – | – | – | – | – |
| Llama-3.2-3B-Instruct | 28.4 | 57.3 | – | – | – | – |

## Project Structure

```
.
├── GenerateSummary.py              # Generates PR summaries via Ollama
├── ACR_ollama.py                   # Automated Code Refinement
├── CTR_ollama.py                   # Change Type Recognition
├── CL_ollama.py                    # Change Localisation (easy + hard)
├── SI_ollama.py                    # Solution Identification (easy + hard)
├── main.py                         # Runs all tasks for both models
├── ollama_api.py                   # HTTP client for the Ollama API
├── utils.py                        # Prompt templates and scoring helpers
├── CodeReviewQA_with_summaries.json # Dataset + generated summaries
├── results/results.csv             # Aggregated scores per model
└── requirements.txt
```

## Reference

```
@inproceedings{lin-etal-2025-codereviewqa,
    title = "{C}ode{R}eview{QA}: The Code Review Comprehension Assessment for Large Language Models",
    author = "Lin, Hong Yi and Liu, Chunhua and Gao, Haoyu and Thongtanunam, Patanamon and Treude, Christoph",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.476/",
    doi = "10.18653/v1/2025.findings-acl.476",
    pages = "9138--9166",
    ISBN = "979-8-89176-256-5"
}
```

Released under the MIT License (inherited from CodeReviewQA).
