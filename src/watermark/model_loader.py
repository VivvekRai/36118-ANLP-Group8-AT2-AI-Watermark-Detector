"""
Loads the base language model used for both watermarked generation and
baseline (unwatermarked) generation.

Kept as its own module so Streamlit app can wrap
`load_model()` in its own caching decorator (`st.cache_resource`) without
this module needing to know anything about the UI framework.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_name: str = "gpt2"):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model