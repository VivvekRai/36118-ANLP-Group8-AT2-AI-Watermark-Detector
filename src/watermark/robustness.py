import random
from typing import Callable, List

from .generator import generate_watermarked_text
from .detector import detect_watermark


def remove_watermark_stub(text: str, drop_prob: float = 0.15) -> str:

    words = text.split()
    kept = [w for w in words if random.random() > drop_prob]
    return " ".join(kept)


def evaluate_robustness(
    tokenizer,
    model,
    prompts: List[str],
    remove_watermark_fn: Callable[[str], str],
    max_new_tokens: int = 40,
) -> List[dict]:

    results = []
    for prompt in prompts:
        original = generate_watermarked_text(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
        attacked = remove_watermark_fn(original)

        original_result = detect_watermark(tokenizer, original)
        attacked_result = detect_watermark(tokenizer, attacked)

        results.append(
            {
                "prompt": prompt,
                "original_z": original_result["z_score"],
                "original_flagged": original_result["is_watermarked"],
                "attacked_z": attacked_result["z_score"],
                "attacked_flagged": attacked_result["is_watermarked"],
            }
        )
    return results