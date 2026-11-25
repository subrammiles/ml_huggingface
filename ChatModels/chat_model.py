"""Chat (instruction-tuned) models and chat templates.

Base models (like GPT-2) only continue text. Instruction-tuned models are
trained on conversations and follow requests. Each model expects the
conversation in its own special format: the CHAT TEMPLATE handles that.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"  # tiny, runs on CPU; quality is basic
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a concise assistant. Answer in one or two sentences."},
    {"role": "user", "content": "What is machine learning?"},
]

# 1) What the template produces: plain text with the model's special markers
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Formatted prompt the model actually sees:\n", prompt_text)
# add_generation_prompt=True appends the "assistant is speaking now" marker.

# 2) Generate a reply
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
torch.manual_seed(0)
output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
reply = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Assistant:", reply)

# 3) Multi-turn: the model has no memory; send the WHOLE conversation each time
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "Give one everyday example of it."})
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
print("\nUser: Give one everyday example of it.")
print("Assistant:", tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# 4) Shortcut: the text-generation pipeline accepts message lists directly
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = chat([{"role": "user", "content": "Name three programming languages."}],
              max_new_tokens=40, do_sample=False)
print("\nPipeline reply:", result[0]["generated_text"][-1]["content"])

# Bigger instruct models (Qwen2.5, Llama 3, Mistral, SmolLM2-1.7B) give much
# better answers but need more memory. Conversations count toward the model's
# context limit, so long chats must be trimmed or summarised.
