"""
Visual proof of watermarking: renders text as HTML with each token
colour-coded by its green/red-list status, so the watermark pattern can be
seen directly rather than only summarized as a z-score.
"""
import html

from .green_list import get_green_list


def highlight_tokens(tokenizer, text: str, green_ratio: float = 0.25, secret_key: int = 15485863) -> str:
    """
    Returns an HTML string with each token wrapped in a coloured <span>:
    green background = token was on the green list (given the previous
    token), red background = token was on the red list. The very first
    token has no "previous token" to seed a green list, so it's shown
    in neutral grey.
    """
    token_ids = tokenizer.encode(text)
    spans = []

    for i, token_id in enumerate(token_ids):
        token_text = html.escape(tokenizer.decode([token_id]))

        if i == 0:
            color = "#888888"  # neutral -- no previous token to seed a green list
        else:
            prev_token_id = token_ids[i - 1]
            green_list = get_green_list(prev_token_id, tokenizer.vocab_size, green_ratio, secret_key)
            color = "#2e7d32" if token_id in green_list else "#8b2e2e"  # green vs red

        spans.append(
            f'<span style="background-color:{color}; color:white; padding:2px 4px; '
            f'margin:1px; border-radius:3px; display:inline-block;">{token_text}</span>'
        )

    return " ".join(spans)