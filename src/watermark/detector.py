"""
Watermark detection: a formal statistical test (z-score for proportions)
answering "is the green-token rate high enough to be very unlikely by
chance?", plus a lighter-weight helper for quick, informal checks.
"""
import math

from .green_list import get_green_list


def check_green_fraction(tokenizer, text: str, green_ratio: float = 0.25, secret_key: int = 15485863) -> float:

    token_ids = tokenizer.encode(text)
    green_hits = 0
    total = 0

    for i in range(1, len(token_ids)):
        prev_token_id = token_ids[i - 1]
        current_token_id = token_ids[i]
        green_list = get_green_list(prev_token_id, tokenizer.vocab_size, green_ratio, secret_key)
        if current_token_id in green_list:
            green_hits += 1
        total += 1

    return green_hits / total if total > 0 else 0.0


def detect_watermark(
    tokenizer,
    text: str,
    green_ratio: float = 0.25,
    secret_key: int = 15485863,
    z_threshold: float = 4.0,
) -> dict:

    token_ids = tokenizer.encode(text)

    green_hits = 0
    total = 0

    for i in range(1, len(token_ids)):
        prev_token_id = token_ids[i - 1]
        current_token_id = token_ids[i]
        green_list = get_green_list(prev_token_id, tokenizer.vocab_size, green_ratio, secret_key)
        if current_token_id in green_list:
            green_hits += 1
        total += 1

    if total == 0:
        return {"error": "No tokens to analyze"}

    expected = total * green_ratio
    std_dev = math.sqrt(total * green_ratio * (1 - green_ratio))
    z_score = (green_hits - expected) / std_dev if std_dev > 0 else 0.0

    is_watermarked = z_score > z_threshold

    return {
        "green_hits": green_hits,
        "total_tokens": total,
        "green_fraction": green_hits / total,
        "z_score": z_score,
        "is_watermarked": is_watermarked,
        "confidence": "high" if z_score > 6 else "moderate" if z_score > z_threshold else "low",
    }