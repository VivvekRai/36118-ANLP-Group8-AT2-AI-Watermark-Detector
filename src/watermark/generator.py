"""
Text generation: watermarked (biased toward the green list at every step)
and baseline (standard, unbiased generation used as a negative control).
"""
import torch

from .green_list import get_green_list


def generate_watermarked_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 40,
    green_ratio: float = 0.25,
    delta: float = 2.0,
    secret_key: int = 15485863,
) -> str:

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]

        prev_token_id = input_ids[0, -1].item()
        green_list = get_green_list(prev_token_id, tokenizer.vocab_size, green_ratio, secret_key)

        green_mask = torch.zeros_like(logits)
        green_indices = torch.tensor(list(green_list))
        green_mask[green_indices] = delta
        biased_logits = logits + green_mask

        probs = torch.softmax(biased_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate_baseline_text(tokenizer, model, prompt: str, max_new_tokens: int = 40) -> str:

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)