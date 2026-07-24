# Do AI-Generated PR Summaries Improve Code Review?

This repository accompanies our investigation into whether automatically generated pull request summaries can improve how LLMs perform on code review tasks. We build on top of the [CodeReviewQA](https://github.com/hongyi-tom/CodeReviewQA) benchmark (Lin et al., ACL 2025) and extend it by injecting AI-generated summaries into the review context, then measuring the effect across multiple comprehension dimensions.

**Research question:** Does providing an LLM with a PR summary alongside the code diff and review comment lead to better code review comprehension?

## How It Works

1. `GenerateSummary.py` produces a PR summary for every example in the CodeReviewQA dataset using Llama 3.2 via a local Ollama instance. The enriched dataset is saved to `CodeReviewQA_with_summaries.json`.

2. The evaluation scripts (`ACR_ollama.py`, `CTR_ollama.py`, `CL_ollama.py`, `SI_ollama.py`) each run a model on the benchmark with and without the summary, then score its output against the gold standard.

3. `main.py` ties everything together -- it loops over both models and all six tasks, skipping anything already recorded in `results/results.csv`.

## Models Under Evaluation

We evaluate two small, open-weight instruct models that are practical to run locally:

| Model | Parameters | Source |
|-------|-----------|--------|
| Qwen2.5-Coder-3B-Instruct | 3B | `qwen2.5-coder:3b` |
| Llama-3.2-3B-Instruct | 3B | `llama3.2:latest` |

Both are served through [Ollama](https://ollama.com/) and called via a lightweight HTTP wrapper (`ollama_api.py`).

## Tasks and Metrics

Each task corresponds to one column in the results table. The first is a generative task (exact match); the remaining five are multiple-choice.

| Abbreviation | Full Name | Type |
|-------------|-----------|------|
| ACR | Automated Code Refinement | Generative (EM%) |
| CTR | Change Type Recognition | MCQ Accuracy% |
| CLE | Change Localisation (Easy) | MCQ Accuracy% |
| CLH | Change Localisation (Hard) | MCQ Accuracy% |
| SIE | Solution Identification (Easy) | MCQ Accuracy% |
| SIH | Solution Identification (Hard) | MCQ Accuracy% |

## Baseline Results (Without Summary)

These are the scores from the original CodeReviewQA setup, where models receive only the code snippet and the review comment -- no summary.

| Model | ACR | CTR | CLE | CLH | SIE | SIH |
|-------|-----|-----|-----|-----|-----|-----|
| Qwen2.5-Coder-3B-Instruct | 30.3 | 77.7 | 1.8 | 1.8 | 12.2 | 8.0 |
| Llama-3.2-3B-Instruct | 25.9 | 78.8 | 0.8 | 0.4 | 9.9 | 7.6 |

## Results With Summary

After running `main.py --summary`, the same models are evaluated with the AI-generated PR summary injected into the prompt. This table will be populated once the full evaluation completes.

| Model | ACR | CTR | CLE | CLH | SIE | SIH |
|-------|-----|-----|-----|-----|-----|-----|
| Qwen2.5-Coder-3B-Instruct | -- | -- | -- | -- | -- | -- |
| Llama-3.2-3B-Instruct | -- | -- | -- | -- | -- | -- |

## Project Structure

```
.
├── GenerateSummary.py          # Generates PR summaries for the dataset via Ollama
├── ACR_ollama.py               # Automated Code Refinement evaluation
├── CTR_ollama.py               # Change Type Recognition evaluation
├── CL_ollama.py                # Change Localisation evaluation (easy + hard)
├── SI_ollama.py                # Solution Identification evaluation (easy + hard)
├── main.py                     # Runs all tasks for both models, writes results to CSV
├── ollama_api.py               # Thin HTTP client for the Ollama API
├── utils.py                    # Prompt templates and evaluation helpers
├── CodeReviewQA_with_summaries.json  # Dataset augmented with generated summaries
├── results/
│   └── results.csv             # Aggregated scores per model
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

You also need [Ollama](https://ollama.com/) running locally with the required models pulled:

```bash
ollama pull llama3.2
ollama pull qwen2.5-coder:3b
```

## Generating Summaries

```bash
python GenerateSummary.py
```

This reads from the Hugging Face dataset `Tomo-Melb/CodeReviewQA`, generates a summary for each example using Llama 3.2, and saves the result to `CodeReviewQA_with_summaries.json`.

## Running Evaluation

```bash
# Run all tasks for both models (with summary)
python main.py

# Or run a single task manually
python ACR_ollama.py llama3.2:latest --summary
python CTR_ollama.py qwen2.5-coder:3b --summary
python CL_ollama.py llama3.2:latest easy --summary
python SI_ollama.py qwen2.5-coder:3b hard --summary
```

## Reference

This project extends the CodeReviewQA benchmark:

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
