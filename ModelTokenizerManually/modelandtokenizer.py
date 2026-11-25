from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Deep learning is"
inputs = tokenizer(input_text, return_tensors="pt")
print("Token IDs:", inputs["input_ids"].tolist())
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,                    # new tokens, not counting the prompt
        do_sample=False,                      # greedy: always the most likely token
        pad_token_id=tokenizer.eos_token_id,  # GPT-2 has no pad token
    )

print("Output token IDs:", outputs[0].tolist())

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
