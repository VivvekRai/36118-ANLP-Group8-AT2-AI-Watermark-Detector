"""
Text generation: watermarked (biased toward the green list at every step)
and baseline (standard, unbiased generation used as a negative control).

GPT-2 can only read 1024 tokens at once (prompt plus everything generated so
far). Both functions therefore trim an over-long prompt, keeping its END, so
that prompt + new tokens always fit in the model's context window.
"""
import torch

from .green_list import get_green_list


def _context_limit(model) -> int:
    """How many tokens the model can read at once (1024 for GPT-2)."""
    return getattr(model.config, "n_positions", 1024)


def _encode_prompt(tokenizer, model, prompt: str, max_new_tokens: int):
    """Encode the prompt and trim it so prompt + max_new_tokens fits the context window."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # An empty prompt has no previous token to build a green list from,
    # so start from the end-of-text token instead.
    if input_ids.shape[1] == 0:
        input_ids = torch.tensor([[tokenizer.eos_token_id]])

    room_for_prompt = max(1, _context_limit(model) - max_new_tokens)
    if input_ids.shape[1] > room_for_prompt:
        input_ids = input_ids[:, -room_for_prompt:]
    return input_ids


def generate_watermarked_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 40,
    green_ratio: float = 0.25,
    delta: float = 2.0,
    secret_key: int = 15485863,
) -> str:

    input_ids = _encode_prompt(tokenizer, model, prompt, max_new_tokens)
    max_context = _context_limit(model)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Sliding window: never feed the model more than it can read.
            # (The green list depends only on the previous token, so the
            # watermark is unaffected.)
            model_input = input_ids[:, -max_context:]
            outputs = model(model_input)
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

    input_ids = _encode_prompt(tokenizer, model, prompt, max_new_tokens)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)