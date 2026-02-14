from transformers import pipeline

classifier = pipeline("zero-shot-classification")

sequence = "I want to learn machine learning."
labels = ["education", "sports", "finance"]

result = classifier(sequence, labels)

print(result)
