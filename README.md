# Hugging Face Transformers

Runnable examples for using pretrained models from the Hugging Face Hub: pipelines, tokenizers, text generation, chat models, embeddings, datasets, and fine-tuning (full and LoRA). Every example runs on a laptop CPU, and the models are small (mostly 100–450 MB).

## Setup

```bash
# CPU-only PyTorch (smaller download); see pytorch.org for GPU builds
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers datasets sentence-transformers peft accelerate
```

Examples are written for **Transformers v5** (tested with 5.24). Models download on first use and are cached in `~/.cache/huggingface` (change it with `HF_HOME`). The first run of each script is slower while it downloads.

The two training scripts write to an `output/` folder in the directory you run them from. That folder is git-ignored.

## Learning path

### 1. Using pretrained models

- `SentimentAnalysis/SentimentAnalysis.py` — the simplest pipeline.
- `Zero-ShotClassification/zeroshotclass.py` — classify into any labels without training.
- `TextGeneration/textgeneration.py` — continue a prompt with GPT-2.
- `PipelinesTour/pipelines_tour.py` — named entities, question answering, fill-mask, multi-label zero-shot, batching.

### 2. What happens inside

- `Tokenization/tokenization.py` — sub-word tokens, IDs, special tokens, padding, attention masks, truncation.
- `ModelTokenizerManually/modelandtokenizer.py` — text → tokens → model → tokens → text, without a pipeline.
- `GenerationStrategies/generation_strategies.py` — next-token probabilities, temperature, greedy, beam search, top-k, top-p, repetition penalty.
- `ChatModels/chat_model.py` — instruction-tuned models, chat templates, multi-turn conversations.

### 3. Embeddings and search

- `EmbeddingsExample/embedding.py` — sentence vectors and cosine similarity.
- `SemanticSearch/semantic_search.py` — masked mean pooling, `sentence-transformers`, searching documents by meaning (the retrieval step of RAG).

### 4. Data and training

- `DatasetsLibrary/datasets_basics.py` — load, explore, filter, map, tokenize, and split datasets.
- `FineTuning/fine_tune_classifier.py` — fine-tune DistilBERT with `Trainer` (about 5 minutes on CPU; accuracy 0.49 → 0.80).
- `Lora_PEFT/lora_fine_tune.py` — LoRA: train ~1% of the weights; the saved adapter is 3 MB instead of 268 MB.

### 5. Practical essentials

- `SaveLoadAndDevices/save_load_devices.py` — GPU/CPU selection, inference mode, batching speed, saving and loading, memory per precision, cache, offline mode, tokens.

## Transformers v5 changes to know

- The `question-answering`, `summarization`, and `translation` pipelines were removed. Use a chat model for summaries and translations; use `AutoModelForQuestionAnswering` directly for extractive QA (see `PipelinesTour/`).
- Prefer `max_new_tokens` over `max_length` in generation.
- Always pass `model=` to `pipeline()`, so results don't change when the library's default model changes.
