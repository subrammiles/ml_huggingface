from transformers import pipeline

# Always name the model: without model=, the pipeline picks a default that can
# change between library versions (and prints a warning).
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

result = classifier("I love learning AI!")

print(result)

# A list of texts is processed as a batch
print(classifier(["The food was cold.", "Best purchase this year!"]))
