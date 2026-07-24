from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

model = "meta/llama-3.2-3b-instruct"

# Load dataset
ds = load_dataset("Tomo-Melb/CodeReviewQA")

data = ds["Benchmark"]

summaries = []

for i, example in enumerate(data):

    old_code = example.get("old", "")
    new_code = example.get("new", "")
    review = example.get("review", "")

    prompt = f"""   
You are a software engineering assistant.

Generate a concise pull request summary based on the code changes and review.

Old code:
{old_code}

New code:
{new_code}

Review:
{review}

Write only the pull request summary.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        summary = response.choices[0].message.content

    except Exception as e:
        summary = f"ERROR: {e}"

    summaries.append(summary)

    print(f"Completed {i+1}/{len(data)}")

# Add summary feature to the dataset
new_dataset = data.add_column(
    "summary",
    summaries
)

# Save the new dataset locally
new_dataset.save_to_disk(
    "CodeReviewQA_with_summaries"
)

print("Dataset saved as CodeReviewQA_with_summaries")

