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
    section("TEST 3: ATTACK COMPARISON (weak vs strong vs compound)")

    attacks = {
        "word deletion (weak)": lambda t: remove_watermark_stub(t, drop_prob=0.15),
        "paraphrase (strong)": lambda t: remove_watermark_paraphrase(t, paraphrase_tokenizer, paraphrase_model),
        "compound (strongest)": lambda t: compound_attack(t, paraphrase_tokenizer, paraphrase_model),
    }

    summary = {name: {"defeated": 0, "drops": [], "similarities": []} for name in attacks}

    for prompt in prompts:
        original = generate_watermarked_text(tokenizer, model, prompt, delta=delta, max_new_tokens=40)
        original_result = detect_watermark(tokenizer, original, z_threshold=z_threshold)

        print(f"\n  Prompt: {prompt[:40]}  (original z={original_result['z_score']:.2f})")

        for name, attack_fn in attacks.items():
            attacked = attack_fn(original)
            attacked_result = detect_watermark(tokenizer, attacked, z_threshold=z_threshold)
            similarity = semantic_similarity(similarity_model, original, attacked)

            drop = original_result["z_score"] - attacked_result["z_score"]
            summary[name]["drops"].append(drop)
            summary[name]["similarities"].append(similarity)
            if not attacked_result["is_watermarked"]:
                summary[name]["defeated"] += 1

            status = "DEFEATED" if not attacked_result["is_watermarked"] else "survived"
            print(f"    {name:<24} after z={attacked_result['z_score']:6.2f}  drop={drop:6.2f}  "
                  f"similarity={similarity:.1%}  [{status}]")

    print("\n  SUMMARY")
    print(f"  {'Attack':<24}{'Defeated':<12}{'Mean drop':<12}{'Mean similarity'}")
    for name, data in summary.items():
        mean_drop = sum(data["drops"]) / len(data["drops"])
        mean_sim = sum(data["similarities"]) / len(data["similarities"])
        defeated_str = f"{data['defeated']}/{len(prompts)}"
        print(f"  {name:<24}{defeated_str:<12}{mean_drop:<12.2f}{mean_sim:.1%}")


def run_edge_cases(tokenizer, model, z_threshold=4.0):
    section("TEST 4: EDGE CASES")
    for label, text in EDGE_CASE_TEXTS:
        print(f"\n  Case: {label!r} -> {text!r}")
        try:
            result = detect_watermark(tokenizer, text, z_threshold=z_threshold)
            if "error" in result:
                print(f"    Handled gracefully: {result['error']}")
            else:
                print(f"    z={result['z_score']:.2f}  green={result['green_fraction']:.1%}  "
                      f"watermarked={result['is_watermarked']}")
        except Exception as e:
            print(f"    CRASHED: {type(e).__name__}: {e}")


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"tests/results_{timestamp}.txt"

    log_file = open(results_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    start = time.time()

    print("Loading GPT-2...")
    tokenizer, model = load_model("gpt2")

    print("Loading paraphraser (humarin/chatgpt_paraphraser_on_T5_base)...")
    paraphrase_tokenizer, paraphrase_model = load_paraphraser()

    print("Loading similarity model (all-MiniLM-L6-v2)...")
    similarity_model = load_similarity_model()

    baseline_results = run_baseline_test(tokenizer, model, BASELINE_PROMPTS)
    run_delta_sweep(tokenizer, model, BASELINE_PROMPTS[:5], DELTA_VALUES)
    run_attack_comparison_test(tokenizer, model, paraphrase_tokenizer, paraphrase_model,
                                similarity_model, BASELINE_PROMPTS[:6])
    run_edge_cases(tokenizer, model)

    elapsed = time.time() - start
    section(f"DONE in {elapsed:.1f} seconds")

    sys.stdout = original_stdout
    log_file.close()
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()