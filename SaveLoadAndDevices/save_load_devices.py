"""Practical essentials: devices, saving/loading, the cache, offline use, and memory.
"""
import os
import time
from pathlib import Path

import torch
from huggingface_hub import constants
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 1) Pick the best available device: NVIDIA GPU, Apple GPU, or CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()  # turns off dropout: always do this before inference

# Inputs must be on the same device as the model
inputs = tokenizer(["What a great day!", "This is awful."], padding=True,
                   return_tensors="pt").to(device)
with torch.inference_mode():  # no gradient tracking: faster, less memory
    logits = model(**inputs).logits
probs = logits.softmax(dim=-1)
for p in probs:
    label = model.config.id2label[int(p.argmax())]
    print(f"  {label} ({p.max():.2f})")

# Pipelines take a device too: pipeline(..., device=device)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

# 2) Batching speeds up inference
texts = ["I really enjoyed this."] * 64
start = time.perf_counter()
for t in texts:
    clf(t)
one_by_one = time.perf_counter() - start
start = time.perf_counter()
clf(texts, batch_size=32)
batched = time.perf_counter() - start
print(f"\n64 texts one by one: {one_by_one:.2f}s, batched: {batched:.2f}s")

# 3) Save and load locally (model and tokenizer together)
save_dir = Path("output/sentiment-model")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("\nSaved files:", sorted(p.name for p in save_dir.iterdir()))
reloaded = AutoModelForSequenceClassification.from_pretrained(save_dir)
print("Reloaded from disk:", type(reloaded).__name__)

# 4) Memory: parameters x bytes per parameter
n_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {n_params / 1e6:.0f}M")
for dtype, nbytes in [("float32", 4), ("float16/bfloat16", 2), ("int8", 1), ("4-bit", 0.5)]:
    print(f"  {dtype:<17} ~{n_params * nbytes / 1e6:.0f} MB")
# Load in half precision on GPU: from_pretrained(name, dtype=torch.bfloat16)
# A 7B model needs ~28 GB in float32 but ~4 GB in 4-bit.

# 5) The cache: downloaded models live here and are reused
print("\nHugging Face cache folder:", constants.HF_HUB_CACHE)
print("Change it with the HF_HOME environment variable.")
# Offline use: after the first download set HF_HUB_OFFLINE=1 to never contact the Hub.
# Pin an exact version of a Hub model with revision="<commit hash>".
# Private / gated models (e.g. Llama): run `huggingface-cli login` or set HF_TOKEN.
# Never hard-code tokens in code; read them from the environment:
print("HF_TOKEN set:", "HF_TOKEN" in os.environ)
