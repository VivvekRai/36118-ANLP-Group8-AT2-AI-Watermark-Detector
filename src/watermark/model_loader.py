from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_name: str = "gpt2"):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model