from transformers import pipeline

# Name the model explicitly. facebook/bart-large-mnli (1.6 GB) is the classic
# choice; this smaller model downloads faster.
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

sequence = "I want to learn machine learning."
labels = ["education", "sports", "finance"]

result = classifier(sequence, labels)

print(result)
print("Best label:", result["labels"][0])
