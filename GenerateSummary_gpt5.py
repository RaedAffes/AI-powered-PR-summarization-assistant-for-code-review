import json

from datasets import load_dataset

ds = load_dataset("Tomo-Melb/CodeReviewQA")
data = ds["Benchmark"]

with open("summaries_gpt5.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

summary_by_index = {s["index"]: s["summary"] for s in summaries}

assert len(summary_by_index) == len(data), "summary count must match dataset size"

records = []
for i, example in enumerate(data):
    assert i in summary_by_index, f"missing summary for index {i}"
    record = dict(example)
    record["summary"] = summary_by_index[i]
    records.append(record)

with open("CodeReviewQA_with_summaries-gpt.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

print(f"Saved {len(records)} records to CodeReviewQA_with_summaries-gpt.json")
