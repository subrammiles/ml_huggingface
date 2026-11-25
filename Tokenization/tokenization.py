"""How tokenizers turn text into numbers, and back.

Models never see text, only token IDs. A tokenizer:
  1. splits text into tokens (often sub-words, so rare words still work)
  2. maps each token to an ID in the model's vocabulary
  3. adds special tokens the model expects (e.g. [CLS] ... [SEP] for BERT)
  4. pads / truncates a batch to one length and builds an attention mask
Each model has its OWN tokenizer: always load it with the same name as the model.
"""
from transformers import AutoTokenizer

bert_tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
gpt2_tok = AutoTokenizer.from_pretrained("openai-community/gpt2")

text = "Tokenizers handle unbelievably rare words."

# 1) Tokens: sub-words for rare words. BERT marks continuations with "##";
#    GPT-2 marks a leading space with "Ġ".
print("BERT tokens:", bert_tok.tokenize(text))
print("GPT-2 tokens:", gpt2_tok.tokenize(text))

# 2) IDs, with special tokens added
encoded = bert_tok(text)
print("\nInput IDs:", encoded["input_ids"])
print("As tokens:", bert_tok.convert_ids_to_tokens(encoded["input_ids"]))
print("Vocabulary size:", bert_tok.vocab_size)

# 3) Decode IDs back to text
print("Decoded:", bert_tok.decode(encoded["input_ids"]))
print("Decoded without special tokens:", bert_tok.decode(encoded["input_ids"], skip_special_tokens=True))

# 4) Batches: texts have different lengths, but a tensor must be rectangular.
batch = ["Short one.", "This sentence is quite a bit longer than the first."]
padded = bert_tok(batch, padding=True, truncation=True, max_length=16, return_tensors="pt")
print("\nPadded input_ids:\n", padded["input_ids"])
print("Attention mask (1 = real token, 0 = padding to ignore):\n", padded["attention_mask"])

# Truncation: texts longer than max_length are cut. Models have a hard limit
# (512 tokens for BERT-style models), so long documents must be split into chunks.
long_text = "word " * 1000
print("\nTokens in a long text:", len(bert_tok(long_text)["input_ids"]))
print("After truncation:", len(bert_tok(long_text, truncation=True)["input_ids"]),
      f"(model max: {bert_tok.model_max_length})")

# Pairs of texts (question + context, sentence A + B) get a separator
pair = bert_tok("What is ML?", "Machine learning learns from data.")
print("\nPair:", bert_tok.decode(pair["input_ids"]))

# Token count matters: API pricing and context limits are measured in tokens.
sentence = "Hugging Face makes transformers easy to use."
print(f"\n'{sentence}' = {len(sentence.split())} words, "
      f"{len(gpt2_tok(sentence)['input_ids'])} GPT-2 tokens")

# GPT-2 has no padding token by default; a common fix for batching:
gpt2_tok.pad_token = gpt2_tok.eos_token
print("GPT-2 pad token set to:", repr(gpt2_tok.pad_token))
