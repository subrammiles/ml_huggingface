"""How text generation picks the next token, and the settings that control it.

A language model outputs a probability for every token in its vocabulary.
Generation repeats: pick a token, append it, predict again.
How we PICK changes the output a lot.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "The best way to learn programming is"
inputs = tokenizer(prompt, return_tensors="pt")

# 1) Look inside: the probabilities for the very next token
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]   # scores for the token after the prompt
probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, 5)
print("Most likely next tokens:")
for p, idx in zip(top.values, top.indices):
    print(f"  {tokenizer.decode(idx)!r:<12} {p:.3f}")

# Temperature reshapes those probabilities before sampling
for t in [0.5, 1.0, 1.5]:
    p_t = torch.softmax(logits / t, dim=-1)
    print(f"temperature {t}: top token probability {p_t.max():.3f}")
# low temperature -> sharper (safer, repetitive); high -> flatter (creative, random)


def generate(label, **settings):
    torch.manual_seed(0)  # makes sampling reproducible
    out = model.generate(**inputs, max_new_tokens=30,
                         pad_token_id=tokenizer.eos_token_id, **settings)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]  # drop the prompt
    print(f"\n[{label}]\n{prompt}{tokenizer.decode(new_tokens, skip_special_tokens=True)}")


# 2) Strategies
generate("greedy: always the most likely token (often repeats itself)", do_sample=False)
generate("beam search: keep the 4 best sequences", do_sample=False, num_beams=4,
         no_repeat_ngram_size=2)
# top_k=0 turns off the default top-k=50 filter, so only temperature acts
generate("sampling, temperature 0.7", do_sample=True, temperature=0.7, top_k=0)
generate("sampling, temperature 1.5 (too random)", do_sample=True, temperature=1.5, top_k=0)
generate("top-k 50: sample only from the 50 likeliest", do_sample=True, top_k=50)
generate("top-p 0.9: smallest set covering 90% probability", do_sample=True, top_p=0.9, top_k=0)
generate("repetition_penalty 1.3", do_sample=False, repetition_penalty=1.3)

# 3) Length settings
#   max_new_tokens: how many tokens to ADD (prefer this)
#   max_length:     total length INCLUDING the prompt (easy to get wrong)
#   eos_token_id / stop strings: stop early when the model signals the end
# Rules of thumb: factual answers -> greedy or low temperature;
# creative writing -> temperature 0.7-1.0 with top_p 0.9.
