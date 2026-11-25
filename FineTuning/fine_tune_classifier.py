"""Fine-tune a pretrained model on your own labelled data with the Trainer.

Transfer learning: start from a model that already understands language
(DistilBERT), add a small classification head, and train briefly on your task.
This needs far less data and compute than training from scratch.

Kept small so it runs on a laptop CPU in a few minutes: 2,000 training examples,
1 epoch. Use the full dataset and a GPU for real results.
"""
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, pipeline)

model_name = "distilbert/distilbert-base-uncased"
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train_ds = ds["train"].shuffle(seed=0).select(range(2_000))
eval_ds = ds["validation"].shuffle(seed=0).select(range(500))

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)


train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

# A pretrained body + a NEW, randomly initialised 2-class head.
# Loading prints a report: "MISSING classifier.*" (the new head, expected) and
# "UNEXPECTED vocab_*" (the pre-training head we do not need). Both are normal here.
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id={v: k for k, v in id2label.items()})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}


args = TrainingArguments(
    output_dir="output/distilbert-rotten-tomatoes",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,          # small: we only nudge the pretrained weights
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    report_to="none",
    seed=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),  # pad each batch to its longest text
    compute_metrics=compute_metrics,
)

print("Accuracy BEFORE training (random head):", trainer.evaluate()["eval_accuracy"])
trainer.train()
print("Accuracy AFTER training:", trainer.evaluate()["eval_accuracy"])

# Save model + tokenizer together, then reload like any Hub model
save_dir = "output/my-sentiment-model"
trainer.save_model(save_dir)
classifier = pipeline("text-classification", model=save_dir)
for review in ["A moving, beautifully acted film.", "Two hours I will never get back."]:
    print(classifier(review)[0], "<-", review)

# Share it: trainer.push_to_hub() after `huggingface-cli login`.
# Next: Lora_PEFT/ trains only a small fraction of the weights.
