"""A tour of ready-made pipelines for common NLP tasks.

pipeline(task, model=...) bundles tokenizer + model + post-processing.
Always pass model= explicitly: the default model can change between versions,
and pinning it makes results reproducible. Models download once, then are cached.

Note for Transformers v5+: the "question-answering", "summarization" and
"translation" pipelines were removed. Summaries and translations are now done
with a chat model (see ChatModels/); question answering is shown below at the
model level.
"""
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Named entity recognition: find people, organisations, places
ner = pipeline("token-classification", model="dslim/bert-base-NER",
               aggregation_strategy="simple")  # merge sub-word pieces into whole words
text = "Sundar Pichai announced the new Google office in Bengaluru on Monday."
print("Named entities:")
for entity in ner(text):
    print(f"  {entity['word']:<15} {entity['entity_group']:<5} score={entity['score']:.2f}")

# Extractive question answering: the model scores every token as a possible
# START and END of the answer; the answer is the span between the best two.
qa_name = "distilbert/distilbert-base-cased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_name)
context = ("Hugging Face was founded in 2016 in New York. Its Transformers library "
           "provides thousands of pretrained models for text, vision and audio.")
for question in ["When was Hugging Face founded?", "What does the library provide?"]:
    inputs = qa_tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        out = qa_model(**inputs)
    start, end = int(out.start_logits.argmax()), int(out.end_logits.argmax())
    answer = qa_tokenizer.decode(inputs["input_ids"][0, start:end + 1])
    print(f"\nQ: {question}\nA: {answer}")

# Fill-mask: what BERT-style models learn during pre-training
fill = pipeline("fill-mask", model="distilbert/distilbert-base-uncased")
print("\nFill in [MASK]:")
for guess in fill("I drink a cup of [MASK] every morning.", top_k=3):
    print(f"  {guess['token_str']:<10} {guess['score']:.3f}")

# Zero-shot classification with a pinned model, and multi-label mode
zero_shot = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
result = zero_shot("The new phone has a great camera but the battery dies quickly.",
                   candidate_labels=["camera", "battery", "price", "shipping"],
                   multi_label=True)  # each label judged on its own
print("\nZero-shot (multi-label):")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:<9} {score:.2f}")

# Batching: pass a list to process many inputs in one call (much faster)
sentiment = pipeline("text-classification",
                     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
reviews = ["Absolutely loved it!", "Terrible, waste of money.", "It was okay, nothing special."]
print("\nBatched sentiment:")
for review, pred in zip(reviews, sentiment(reviews, batch_size=8)):
    print(f"  {pred['label']:<8} {pred['score']:.2f}  {review}")
# Note the last one: a 2-class model must pick POSITIVE or NEGATIVE even for
# neutral text. Check what labels a model was trained on before using it.

# Other pipeline tasks: "image-classification", "automatic-speech-recognition",
# "object-detection", "feature-extraction", "image-text-to-text", ...
# List them with: from transformers.pipelines import SUPPORTED_TASKS
# Browse models per task at https://huggingface.co/models
