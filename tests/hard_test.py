"""
Hard testing script: runs the full pipeline (generate, detect, and three
tiers of attack) across many trials to produce statistics for the report,
rather than relying on one-off manual UI clicks.

Every run automatically saves a timestamped results file to tests/, in
addition to printing to the terminal.

Run from the project root with: python tests\\hard_test.py
"""
import datetime
import sys
import time

sys.path.insert(0, ".")

from src.watermark import load_model, generate_watermarked_text, detect_watermark
from src.watermark.removal import load_paraphraser, remove_watermark_paraphrase, compound_attack
from src.watermark.robustness import remove_watermark_stub
from src.watermark.similarity import load_similarity_model, semantic_similarity


class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


BASELINE_PROMPTS = [
    "The weather today is",
    "In recent economic news,",
    "Scientists have discovered that",
    "The history of Rome shows",
    "Cooking a good meal requires",
    "Modern architecture often features",
    "The stock market reacted to",
    "Education systems around the world",
    "Climate change affects",
    "The novel begins with",
]

DELTA_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0]

EDGE_CASE_TEXTS = [
    ("empty string", ""),
    ("single word", "Hello"),
    ("very short", "AI is good."),
    ("repeated word", "the the the the the the the the the the"),
    ("numbers only", "12345 67890 11111 22222"),
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_baseline_test(tokenizer, model, prompts, z_threshold=4.0):
    section("TEST 1: FALSE POSITIVE RATE (unwatermarked baseline text)")
    results = []
    for prompt in prompts:
        text = generate_watermarked_text(tokenizer, model, prompt, delta=0.0, max_new_tokens=40)
        result = detect_watermark(tokenizer, text, z_threshold=z_threshold)
        results.append(result["z_score"])
        flag = "FALSE POSITIVE" if result["is_watermarked"] else "ok"
        print(f"  z={result['z_score']:6.2f}  green={result['green_fraction']:.1%}  [{flag}]  {prompt[:40]}")

    false_positives = sum(1 for z in results if z > z_threshold)
    mean_z = sum(results) / len(results)
    print(f"\n  Mean z-score: {mean_z:.2f}")
    print(f"  False positives: {false_positives}/{len(results)}")
    return results


def run_delta_sweep(tokenizer, model, prompts, deltas, z_threshold=4.0):
    section("TEST 2: DELTA SWEEP (watermark strength vs detectability)")
    print(f"  {'Delta':<8}{'Mean Z':<10}{'Detected':<12}{'Min Z':<10}{'Max Z':<10}")
    for delta in deltas:
        z_scores = []
        for prompt in prompts:
            text = generate_watermarked_text(tokenizer, model, prompt, delta=delta, max_new_tokens=40)
            result = detect_watermark(tokenizer, text, z_threshold=z_threshold)
            z_scores.append(result["z_score"])
        mean_z = sum(z_scores) / len(z_scores)
        detected = sum(1 for z in z_scores if z > z_threshold)
        print(f"  {delta:<8}{mean_z:<10.2f}{f'{detected}/{len(z_scores)}':<12}{min(z_scores):<10.2f}{max(z_scores):<10.2f}")


def run_attack_comparison_test(tokenizer, model, paraphrase_tokenizer, paraphrase_model,
                                similarity_model, prompts, delta=2.0, z_threshold=4.0):
    section("TEST 3: ATTACK COMPARISON (deletion vs paraphrase vs compound)")

    attacks = {
        "word deletion": lambda t: remove_watermark_stub(t, drop_prob=0.15),
        "paraphrase": lambda t: remove_watermark_paraphrase(t, paraphrase_tokenizer, paraphrase_model),
        "compound