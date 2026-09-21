"""
Real watermark-removal attacks: paraphrases text using a small T5-based
paraphrasing model, and optionally stacks a synonym-substitution pass on
top for a stronger, compound attack.

Paraphrasing alone is a much more realistic attack than random word
deletion -- it rewrites whole clauses, which is exactly what actually
defeats green/red-list watermarks in practice (see Kirchenbauer et al.'s
own follow-up work on watermark robustness, 2024). Stacking a synonym
swap on top compounds the disruption further.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .synonym_attack import synonym_substitute


def load_paraphraser():
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model.eval()
    return tokenizer, model


def remove_watermark_paraphrase(text, paraphrase_tokenizer, paraphrase_model, max_length=128):
    input_ids = paraphrase_tokenizer(
        f"paraphrase: {text}",
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = paraphrase_model.generate(
        input_ids,
        do_sample=True,
        num_return_sequences=1,
        repetition_penalty=10.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        max_length=max_length,
    )

    return paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)


def compound_attack(text, paraphrase_tokenizer, paraphrase_model, swap_prob=0.3):
    """
    Strongest attack available here: paraphrase first (rewrites sentence
    structure), then synonym-swap the result (further disrupts individual
    token choices). Stacking attacks compounds the damage to the
    watermark's green/red pattern beyond either attack alone.
    """
    paraphrased = remove_watermark_paraphrase(text, paraphrase_tokenizer, paraphrase_model)
    return synonym_substitute(paraphrased, swap_prob=swap_prob)