r"""
Pools the saved hard_test runs into the numbers for the report tables.

Run from the project root after the hard_test runs:
    python tests\summarise_runs.py
It reads every tests/results_*_seed*.json file.
"""
import glob
import json
import math
import sys


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    files = sorted(glob.glob("tests/results_*_seed*.json"))
    if not files:
        print("No tests/results_*_seed*.json files found. Run tests/hard_test.py first.")
        sys.exit(1)

    runs = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            runs.append((path, json.load(f)))

    print(f"Pooling {len(runs)} runs:")
    for path, data in runs:
        env = data["meta"]["environment"]
        print(f"  {path}  seed={data['meta']['seed']}  commit={env.get('git_commit')}"
              f"  uncommitted={env.get('git_uncommitted_changes')}  {data['meta']['seconds']}s")
    commits = {d["meta"]["environment"].get("git_commit") for _, d in runs}
    if len(commits) > 1:
        print(f"  WARNING: runs came from different commits: {commits}")

    # ---- Table 6: control and delta sweep ----
    print("\nTABLE 6: detection by delta (pooled)")
    print(f"  {'delta':<12}{'n':<6}{'mean z':<10}{'min z':<10}{'max z':<10}{'detected'}")
    control = [r for _, d in runs for r in d["control"] if "z" in r]
    zs = [r["z"] for r in control]
    det = sum(1 for r in control if r["detected"])
    print(f"  {'0 (control)':<12}{len(zs):<6}{mean(zs):<10.2f}{min(zs):<10.2f}{max(zs):<10.2f}{det} / {len(zs)}")
    deltas = sorted({r["delta"] for _, d in runs for r in d["sweep"]})
    for delta in deltas:
        rs = [r for _, d in runs for r in d["sweep"] if r["delta"] == delta and "z" in r]
        zs = [r["z"] for r in rs]
        det = sum(1 for r in rs if r["detected"])
        print(f"  {delta:<12}{len(zs):<6}{mean(zs):<10.2f}{min(zs):<10.2f}{max(zs):<10.2f}{det} / {len(zs)}")

    # ---- Table 5: attacks ----
    attacks = [r for _, d in runs for r in d["attacks"]]
    originals = {}
    for _, d in runs:
        for r in d["attacks"]:
            originals[(d["meta"]["seed"], r["prompt"])] = r["original"]
    oz = [o["z"] for o in originals.values()]
    n_det = sum(1 for o in originals.values() if o["detected"])
    print(f"\nTABLE 5: attack comparison at delta = 2.0")
    print(f"  Originals: {len(oz)}  detected: {n_det}  mean z {mean(oz):.2f}  range {min(oz):.2f} to {max(oz):.2f}")
    print(f"  Mean tokens per original: {mean([o['tokens'] for o in originals.values()]):.1f}")
    print(f"  {'attack':<28}{'defeated':<12}{'mean z after':<14}{'max z after':<13}{'mean drop':<11}"
          f"{'mean sim':<10}{'sim range':<14}{'mean tokens'}")
    names = []
    for r in attacks:
        if r["attack"] not in names:
            names.append(r["attack"])
    for name in names:
        rs = [r for r in attacks if r["attack"] == name]
        elig = [r for r in rs if r["eligible"]]
        defeated = sum(1 for r in elig if r["defeated"])
        after = [r["attacked"]["z"] for r in rs]
        drops = [r["original"]["z"] - r["attacked"]["z"] for r in rs]
        sims = [r["similarity"] for r in rs if r["similarity"] is not None]
        toks = [r["attacked"]["tokens"] for r in rs]
        rng = f"{min(sims):.2f}-{max(sims):.2f}" if sims else "n/a"
        print(f"  {name:<28}{f'{defeated} / {len(elig)}':<12}{mean(after):<14.2f}{max(after):<13.2f}"
              f"{mean(drops):<11.2f}{mean(sims):<10.3f}{rng:<14}{mean(toks):.1f}")

    # ---- Paired comparison: does the synonym step add anything? ----
    by_key = {}
    for r in attacks:
        by_key.setdefault((r["original_text"]), {})[r["attack"]] = r
    pairs = [(v["paraphrase"], v["compound_same_paraphrase"]) for v in by_key.values()
             if "paraphrase" in v and "compound_same_paraphrase" in v]
    if pairs:
        z_diff = [c["attacked"]["z"] - p["attacked"]["z"] for p, c in pairs]
        s_diff = [c["similarity"] - p["similarity"] for p, c in pairs
                  if c["similarity"] is not None and p["similarity"] is not None]
        lower = sum(1 for d in z_diff if d < 0)
        print(f"\nSYNONYM STEP ON THE SAME PARAPHRASE ({len(pairs)} pairs)")
        print(f"  Mean change in z when synonyms are added: {mean(z_diff):+.2f}  "
              f"(z lower in {lower} of {len(pairs)} pairs)")
        print(f"  Mean change in similarity: {mean(s_diff):+.3f}")

    # ---- Per-run detection at delta 2.0 ----
    print("\nDETECTED AT DELTA 2.0, PER RUN (first 5 prompts, sweep)")
    for path, d in runs:
        rs = [r for r in d["sweep"] if r["delta"] == 2.0]
        print(f"  seed {d['meta']['seed']}: {sum(1 for r in rs if r['detected'])} of {len(rs)}")

    # ---- Edge cases ----
    print("\nEDGE CASES (last run)")
    for r in runs[-1][1]["edge"]:
        if "crashed" in r:
            print(f"  {r['case']}: CRASHED {r['crashed']}")
        elif "error" in r:
            print(f"  {r['case']}: handled ({r['error']})")
        else:
            print(f"  {r['case']}: z={r['z']:.2f} watermarked={r['detected']}")

    # ---- Lower bound for the paraphrase defeat rate ----
    para = [r for r in attacks if r["attack"] == "paraphrase" and r["eligible"]]
    if para and all(r["defeated"] for r in para):
        n = len(para)
        print(f"\n95% one-sided lower bound if all {n} are defeated: {0.05 ** (1 / n):.1%}")


if __name__ == "__main__":
    main()
