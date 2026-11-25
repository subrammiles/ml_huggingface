from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

texts = ["Machine learning is amazing", "I enjoy studying AI", "The weather is cold today"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Mean pooling over real tokens only: the attention mask hides padding
mask = inputs["attention_mask"].unsqueeze(-1).float()
embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

# Normalize to length 1 so a dot product equals cosine similarity
embeddings = F.normalize(embeddings, dim=1)

print("Embedding shape:", embeddings.shape)  # (3 texts, 384 numbers each)
print("Similarity of text 1 to text 2:", round((embeddings[0] @ embeddings[1]).item(), 3))
print("Similarity of text 1 to text 3:", round((embeddings[0] @ embeddings[2]).item(), 3))
# See SemanticSearch/ for searching documents with embeddings.
