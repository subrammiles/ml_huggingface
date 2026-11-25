"""Sentence embeddings and semantic search: find text by MEANING, not exact words.

An embedding model turns each text into a vector. Similar meanings give
vectors pointing in similar directions (high cosine similarity).
This is the retrieval step of RAG systems.
"""
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# 1) Doing it with plain transformers: masked mean pooling + normalization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
sentences = ["A cat sits on the mat.", "A kitten is resting on a rug.", "Stock prices fell today."]
batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    token_vectors = model(**batch).last_hidden_state        # (batch, tokens, 384)
mask = batch["attention_mask"].unsqueeze(-1).float()        # ignore padding tokens
sentence_vectors = (token_vectors * mask).sum(1) / mask.sum(1)
sentence_vectors = F.normalize(sentence_vectors, dim=1)     # length 1 -> dot = cosine
print("Embedding shape:", tuple(sentence_vectors.shape))
print("Similarity matrix (cosine):\n", (sentence_vectors @ sentence_vectors.T).round(decimals=2))
# Without the mask, padding tokens would be averaged in and change the vectors.

# 2) The sentence-transformers library does all of that in one line
st_model = SentenceTransformer(model_name)
same = st_model.encode(sentences, normalize_embeddings=True)
print("Matches the manual version:",
      torch.allclose(torch.tensor(same), sentence_vectors, atol=1e-4))

# 3) Semantic search over a small knowledge base
documents = [
    "To reset your password, click 'Forgot password' on the login page.",
    "Our office is open Monday to Friday, 9am to 6pm.",
    "Refunds are processed within 5 business days after we receive the item.",
    "You can change your delivery address from the Orders page before shipping.",
    "We accept credit cards, UPI, and net banking.",
    "Premium members get free shipping on all orders.",
]
doc_vectors = st_model.encode(documents, normalize_embeddings=True)

queries = ["I forgot my login details", "when will I get my money back",
           "can I pay with a card", "what time do you close"]
for query in queries:
    q_vec = st_model.encode(query, normalize_embeddings=True)
    scores = doc_vectors @ q_vec
    best = scores.argsort()[::-1][:2]
    print(f"\nQuery: {query}")
    for i in best:
        print(f"  {scores[i]:.2f}  {documents[i]}")
# The queries share almost no words with the matching documents. Keyword
# search would miss them; embeddings match on meaning.

# For many documents, store vectors in a vector database (FAISS, Chroma,
# pgvector, ...) instead of comparing against every vector.
# Embedding models differ in language support, max length, and quality:
# compare them on the MTEB leaderboard.
