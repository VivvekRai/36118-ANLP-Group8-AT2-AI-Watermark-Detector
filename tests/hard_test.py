import argparse
import datetime
import json
import platform
import random
import subprocess
import sys
import time

sys.path.insert(0, ".")

import torch

from src.watermark import load_model, generate_watermarked_text, detect_watermark
from src.watermark.removal import load_paraphraser, remove_watermark_paraphrase, compound_attack
import src.watermark.removal as removal_module
from src.watermark.robustness import remove_watermark_stub
from src.watermark.similarity import load_similarity_model, semantic_similarity
from src.watermark.synonym_attack import synonym_substitute


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
GEN_TOKENS = 40
Z_THRESHOLD = 4.0
SWAP_PROB = 0.3

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


def seed_everything(seed):

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    syn_rng = random.Random(seed)

    # compound_attack calls synonym_substitute without a seed, which would use
    # OS randomness. Wrap it so the seed flows from our RNG instead.
    original = getattr(removal_module, "synonym_substitute", None)
    if original is not None:
        def seeded(text, swap_prob=0.3, seed=None):
            use = syn_rng.randrange(2 ** 31) if seed is None else seed
            return original(text, swap_prob=swap_prob, seed=use)
        removal_module.synonym_substitute = seeded
        print(f"  Seeded synonym step inside compound_attack: yes")
    else:
        print("  WARNING: removal.synonym_substitute not found; compound_attack synonym step is NOT seeded")
    return syn_rng


def analyse(tokenizer, text):

    n_tokens = len(tokenizer.encode(text)) if text else 0
    result = detect_watermark(tokenizer, text, z_threshold=Z_THRESHOLD)
    if "error" in result:
        return {"error": result["error"], "tokens": n_tokens}
    return {
        "z": float(result["z_score"]),
        "green_fraction": float(result["green_fraction"]),
        "detected": bool(result["is_watermarked"]),
        "tokens": n_tokens,
    }


def run_baseline_test(tokenizer, model, prompts):
    section("TEST 1: FALSE POSITIVE RATE (delta = 0, no watermark)")
    records = []
    for prompt in prompts:
        text = generate_watermarked_text(tokenizer, model, prompt, delta=0.0, max_new_tokens=GEN_TOKENS)
        a = analyse(tokenizer, text)
        records.append({"prompt": prompt, "delta": 0.0, "text": text, **a})
        flag = "FALSE POSITIVE" if a.get("detected") else "ok"
        print(f"  z={a['z']:6.2f}  green={a['green_fraction']:.1%}  tokens={a['tokens']}  [{flag}]  {prompt[:40]}")

    zs = [r["z"] for r in records if "z" in r]
    print(f"\n  Mean z-score: {sum(zs) / len(zs):.2f}")
    print(f"  False positives: {sum(1 for r in records if r.get('detected'))}/{len(records)}")
    return records


def run_delta_sweep(tokenizer, model, prompts, deltas):
    section("TEST 2: DELTA SWEEP (watermark strength vs detectability)")
    print(f"  {'Delta':<8}{'Mean Z':<10}{'Detected':<12}{'Min Z':<10}{'Max Z':<10}")
    records = []
    for delta in deltas:
        zs = []
        detected = 0
        for prompt in prompts:
            text = generate_watermarked_text(tokenizer, model, prompt, delta=delta, max_new_tokens=GEN_TOKENS)
            a = analyse(tokenizer, text)
            records.append({"prompt": prompt, "delta": delta, "text": text, **a})
            zs.append(a["z"])
            detected += int(a["detected"])
        print(f"  {delta:<8}{sum(zs) / len(zs):<10.2f}{f'{detected}/{len(zs)}':<12}{min(zs):<10.2f}{max(zs):<10.2f}")
    return records


def run_attack_comparison_test(tokenizer, model, para_tok, para_model, sim_model, prompts, syn_rng, delta=2.0):
    section("TEST 3: ATTACK COMPARISON")
    print("  deletion  = drop each word with probability 0.15")
    print("  paraphrase = T5 paraphraser only")
    print("  compound = the app's compound_attack (its own paraphrase, then synonyms)")
    print("  compound_same_paraphrase = synonyms applied to the SAME paraphrase used above")
    print("    (the controlled comparison: it isolates what the synonym step adds)")

    records = []
    names = ["deletion", "paraphrase", "compound", "compound_same_paraphrase"]
    stats = {n: {"eligible": 0, "defeated": 0, "drops": [], "sims": [], "after": []} for n in names}
    skipped = 0

    for prompt in prompts:
        original = generate_watermarked_text(tokenizer, model, prompt, delta=delta, max_new_tokens=GEN_TOKENS)
        orig = analyse(tokenizer, original)
        print(f"\n  Prompt: {prompt[:40]}  (original z={orig['z']:6.2f}, tokens={orig['tokens']})")

        para = remove_watermark_paraphrase(original, para_tok, para_model)
        attacked_texts = {
            "deletion": remove_watermark_stub(original, drop_prob=0.15),
            "paraphrase": para,
            "compound": compound_attack(original, para_tok, para_model),
            "compound_same_paraphrase": synonym_substitute(para, swap_prob=SWAP_PROB, seed=syn_rng.randrange(2 ** 31)),
        }

        for name in names:
            attacked = attacked_texts[name]
            att = analyse(tokenizer, attacked)
            if "error" in att:
                print(f"    {name:<26} skipped (attacked text too short: {att['error']})")
                continue
            try:
                sim = float(semantic_similarity(sim_model, original, attacked))
            except Exception:
                sim = None

            drop = orig["z"] - att["z"]
            eligible = orig["detected"]
            status = "N/A (original not detected)" if not eligible else ("DEFEATED" if not att["detected"] else "survived")
            sim_txt = f"{sim:.1%}" if sim is not None else "n/a"
            print(f"    {name:<26} after z={att['z']:6.2f}  drop={drop:6.2f}  similarity={sim_txt}  tokens={att['tokens']}  [{status}]")

            s = stats[name]
            s["after"].append(att["z"])
            s["drops"].append(drop)
            if sim is not None:
                s["sims"].append(sim)
            if eligible:
                s["eligible"] += 1
                if not att["detected"]:
                    s["defeated"] += 1

            records.append({
                "prompt": prompt, "delta": delta, "attack": name,
                "original_text": original, "attacked_text": attacked,
                "original": orig, "attacked": att, "similarity": sim,
                "eligible": eligible, "defeated": bool(eligible and not att["detected"]),
            })
        if not orig["detected"]:
            skipped += 1

    print("\n  SUMMARY (defeats are counted only over originals that were detected)")
    print(f"  Originals not detected before any attack: {skipped}/{len(prompts)}")
    print(f"  {'Attack':<26}{'Defeated':<12}{'Mean z after':<14}{'Mean drop':<12}{'Mean similarity'}")
    for name, s in stats.items():
        if not s["drops"]:
            continue
        sim_mean = sum(s["sims"]) / len(s["sims"]) if s["sims"] else float("nan")
        print(f"  {name:<26}{f'{s['defeated']}/{s['eligible']}':<12}"
              f"{sum(s['after']) / len(s['after']):<14.2f}{sum(s['drops']) / len(s['drops']):<12.2f}{sim_mean:.1%}")
    return records


def run_edge_cases(tokenizer):
    section("TEST 4: EDGE CASES")
    records = []
    for label, text in EDGE_CASE_TEXTS:
        print(f"\n  Case: {label!r} -> {text!r}")
        try:
            a = analyse(tokenizer, text)
            records.append({"case": label, "text": text, **a})
            if "error" in a:
                print(f"    Handled gracefully: {a['error']}")
            else:
                print(f"    z={a['z']:.2f}  green={a['green_fraction']:.1%}  watermarked={a['detected']}")
        except Exception as e:
            records.append({"case": label, "text": text, "crashed": f"{type(e).__name__}: {e}"})
            print(f"    CRASHED: {type(e).__name__}: {e}")
    return records


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    for pkg in ("transformers", "sentence_transformers", "nltk", "streamlit"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = "unknown"
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = [line for line in subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()
                 if "tests/results_" not in line]
        info["git_uncommitted_changes"] = bool(dirty)
    except Exception:
        info["git_commit"] = "unknown"
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"tests/results_{timestamp}_seed{args.seed}"

    log_file = open(base + ".txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    start = time.time()
    env = environment_info()
    print(f"Seed: {args.seed}")
    for k, v in env.items():
        print(f"  {k}: {v}")

    print("Loading GPT-2...")
    tokenizer, model = load_model("gpt2")
    print("Loading paraphraser (humarin/chatgpt_paraphraser_on_T5_base)...")
    para_tok, para_model = load_paraphraser()
    print("Loading similarity model (all-MiniLM-L6-v2)...")
    sim_model = load_similarity_model()

    syn_rng = seed_everything(args.seed)

    output = {"meta": {"seed": args.seed, "timestamp": timestamp, "gen_tokens": GEN_TOKENS,
                       "z_threshold": Z_THRESHOLD, "environment": env}}
    output["control"] = run_baseline_test(tokenizer, model, BASELINE_PROMPTS)
    output["sweep"] = run_delta_sweep(tokenizer, model, BASELINE_PROMPTS[:5], DELTA_VALUES)
    output["attacks"] = run_attack_comparison_test(tokenizer, model, para_tok, para_model, sim_model,
                                                   BASELINE_PROMPTS[:6], syn_rng)
    output["edge"] = run_edge_cases(tokenizer)

    elapsed = time.time() - start
    output["meta"]["seconds"] = round(elapsed, 1)
    section(f"DONE in {elapsed:.1f} seconds")

    sys.stdout = original_stdout
    log_file.close()
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {base}.txt and {base}.json")


if __name__ == "__main__":
    main()