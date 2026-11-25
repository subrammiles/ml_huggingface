from transformers import pipeline

generator = pipeline("text-generation", model="openai-community/gpt2")

result = generator(
    "Machine learning is",
    max_new_tokens=50,          # tokens to ADD after the prompt
    num_return_sequences=1,
    do_sample=True,             # sample for varied text (see GenerationStrategies/)
    pad_token_id=generator.tokenizer.eos_token_id,
)

print(result[0]["generated_text"])
