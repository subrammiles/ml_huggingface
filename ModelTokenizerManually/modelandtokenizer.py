from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Deep learning is"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=40)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
