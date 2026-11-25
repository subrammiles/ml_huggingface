"""The datasets library: load, explore, clean, tokenize, and split data.

Datasets are stored on disk (Apache Arrow) and loaded lazily, so even very
large datasets do not need to fit in memory. map() caches its results.
"""
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# 1) Load a dataset from the Hub (small: ~10k movie-review sentences)
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
print(ds)  # a DatasetDict with train / validation / test splits
train = ds["train"]
print("\nFeatures:", train.features)
print("First example:", train[0])
print("Label names:", train.features["label"].names)

# 2) Explore: columns, slices, and pandas
print("\nFirst 3 labels:", train["label"][:3])
df = train.to_pandas()
print("Label balance:\n", df["label"].value_counts())
print("Average length (characters):", round(df["text"].str.len().mean()))

# 3) Filter and add columns with map
short = train.filter(lambda ex: len(ex["text"]) < 60)
print(f"\nShort reviews: {len(short)} of {len(train)}")
with_length = train.map(lambda ex: {"n_words": len(ex["text"].split())})
print("New column:", with_length.column_names)

# 4) Tokenize the whole dataset (batched=True is much faster)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenized = train.map(lambda batch: tokenizer(batch["text"], truncation=True),
                      batched=True)
print("\nAfter tokenizing:", tokenized.column_names)
print("Token IDs of first review:", tokenized[0]["input_ids"][:12], "...")

# 5) Shuffle, select, and split
subset = train.shuffle(seed=42).select(range(1_000))
splits = subset.train_test_split(test_size=0.2, seed=42)
print("\nCustom split:", {name: len(part) for name, part in splits.items()})

# 6) Your own data: from a dict, pandas, CSV or JSON files
own = Dataset.from_dict({"text": ["great product", "broke after a day"], "label": [1, 0]})
print("\nFrom a dict:", own)
# Dataset.from_pandas(df), load_dataset("csv", data_files="reviews.csv"),
# load_dataset("json", data_files="data.jsonl")

# Very large datasets: load_dataset(..., streaming=True) reads examples one by
# one without downloading everything first.
