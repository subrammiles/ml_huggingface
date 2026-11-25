"""Parameter-efficient fine-tuning (PEFT) with LoRA.

Full fine-tuning updates every weight: big models need lots of GPU memory,
and every fine-tuned copy is as large as the original.
LoRA freezes the original weights and trains small low-rank "adapter"
matrices added to some layers. Typically <1-2% of the parameters are trained,
and the saved adapter is a few MB instead of hundreds.
"""
import os

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train_ds = ds["train"].shuffle(seed=0).select(range(2_000))
eval_ds = ds["validation"].shuffle(seed=0).select(range(500))
train_ds = train_ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=128), batched=True)
eval_ds = eval_ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=128), batched=True)

base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,        # keeps the new classification head trainable
    r=8,                               # rank of the adapter matrices (size/capacity)
    lora_alpha=16,                     # scaling of the adapter's contribution
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"], # which layers get adapters (attention query/value)
)
model = get_peft_model(base, lora_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="output/lora-run",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=5e-4,            # LoRA usually uses a higher learning rate
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=0,
    ),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
print("LoRA accuracy:", trainer.evaluate()["eval_accuracy"])

# Save ONLY the adapter: tiny compared with the full model
adapter_dir = "output/lora-adapter"
model.save_pretrained(adapter_dir)
size_mb = sum(os.path.getsize(os.path.join(adapter_dir, f)) for f in os.listdir(adapter_dir)) / 1e6
print(f"Adapter size on disk: {size_mb:.1f} MB (the base model is about 268 MB)")

# Load later: base model + adapter. merge_and_unload() bakes the adapter into
# the weights for faster inference.
reloaded = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2), adapter_dir)
merged = reloaded.merge_and_unload()
print("Merged model type:", type(merged).__name__)

# For large language models, QLoRA combines LoRA with 4-bit quantization
# (bitsandbytes, needs a GPU) so 7B+ models can be fine-tuned on one GPU.
