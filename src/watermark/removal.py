"""
Real watermark-removal attacks: paraphrases text using a small T5-based
paraphrasing model, and optionally stacks a synonym-substitution pass on
top for a compound attack.

Paraphrasing alone is a much more realistic attack than random word
deletion -- it rewrites whole clauses, which is exactly what actually
defeats green/red-list watermarks in practice (see Kirchenbauer et al.'s
own follow-up work on watermark robustness, 2024). A synonym swap on top
disrupts individual word choices further, at some extra cost in meaning.

Long text is split into chunks of whole sentences and paraphrased chunk by
chunk, because the paraphraser can only read and write about 128 tokens at
a time.
"""
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .synonym_attack import synonym_substitute

# Roughly how many tokens of source text go into each paraphrase call.
# Kept well under the 128-token limit to leave room for the "paraphrase: " prefix.
CHUNK_TOKENS = 80


def load_paraphraser():
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model.eval()
    return tokenizer, model


def _paraphrase_chunk(text, paraphrase_tokenizer, paraphrase_model, max_length=128):
    """Paraphrase one short piece of text (up to about max_length tokens)."""
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


def _split_into_chunks(text, tokenizer, max_tokens=CHUNK_TOKENS):
    """Group whole sentences into chunks of roughly max_tokens tokens each."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if not sentence:
            continue
        n_tokens = len(tokenizer(sentence, add_special_tokens=False).input_ids)
        if current and current_len + n_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def remove_watermark_paraphrase(text, paraphrase_tokenizer, paraphrase_model, max_length=128):
    """
    Paraphrase text of any length by processing it chunk by chunk, so nothing
    past the first ~128 tokens is silently cut off.

    Limitation: a single sentence longer than about 128 tokens is still
    truncated, which is rare in normal prose.
    """
    chunks = _split_into_chunks(text, paraphrase_tokenizer)
    if len(chunks) <= 1:
        return _paraphrase_chunk(text, paraphrase_tokenizer, paraphrase_model, max_length)
    return " ".join(
        _paraphrase_chunk(chunk, paraphrase_tokenizer, paraphrase_model, max_length)
        for chunk in chunks
    )


def compound_attack(text, paraphrase_tokenizer, paraphrase_model, swap_prob=0.3):
    """
    Compound attack: paraphrase first (rewrites sentence structure), then
    synonym-swap the result (further disrupts individual token choices).

    In our recorded tests the synonym step lowered semantic similarity
    without reliably lowering the z-score more than paraphrasing alone, so
    this is a "paraphrase plus extra word-level noise" attack, not a
    guaranteed stronger one.
    """
    paraphrased = remove_watermark_paraphrase(text, paraphrase_tokenizer, paraphrase_model)
    return synonym_substitute(paraphrased, swap_prob=swap_prob)