"""
Loads the base language model used for both watermarked generation and
baseline (unwatermarked) generation.

Kept as its own module so a caller (e.g. a Streamlit app) can wrap
`load_model()` in its own caching decorator (`st.cache_resource`) without
this module needing to know anything about the UI framework.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_name: str = "gpt2"):
    """
    Load a GPT-2-family tokenizer and model.

    Defaults to full `gpt2` (rather than a distilled variant) to match the
    model used for all validated testing and results (Kaggle notebook:
    z-scores, false-positive rates, parameter sweep).

    Returns
    -------
    tokenizer, model
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model