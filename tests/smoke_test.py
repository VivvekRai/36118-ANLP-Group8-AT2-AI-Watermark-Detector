r"""
Smoke tests for the fixes. Run from the project root:

    python tests\smoke_test.py

Part A checks the logic directly. Part B drives the real Streamlit app headlessly.
"""
import os
import re
import sys

sys.path.insert(0, ".")

from src.watermark import load_model, generate_watermarked_text, detect_watermark
from src.watermark.detector import evidence_strength
from src.watermark.generator import generate_baseline_text
from src.watermark.removal import (
    load_paraphraser, compound_attack, remove_watermark_paraphrase,
    _paraphrase_chunk, _split_into_chunks,
)
from src.watermark.synonym_attack import synonym_substitute

RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append((name, bool(passed)))
    tag = "PASS" if passed else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{extra}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


LONG_TEXT = " ".join([
    "The city council approved a new plan to expand the public transport network next year.",
    "Engineers will add three light rail lines that connect the outer suburbs to the central business district.",
    "Residents have asked for cheaper tickets and more frequent services during the early morning.",
    "A small team of researchers studied how commuters choose between buses, trains and bicycles.",
    "They found that reliability matters more than speed for most people who travel every day.",
    "The report also showed that safe walking paths encourage families to leave their cars at home.",
    "Local businesses expect more customers once the new stations open near the main shopping streets.",
    "Some neighbours worry about noise during construction, so the council promised regular community meetings.",
    "The budget includes funding for solar panels on station roofs and for electric buses.",
    "Training programs will help workers move from older industries into the growing transport sector.",
    "Planners say the project could cut traffic jams by almost a fifth within ten years.",
    "Critics argue that the money would be better spent on repairing existing roads and bridges.",
    "Supporters respond that a modern network will attract new residents and keep young people in town.",
    "An independent committee will review progress every six months and publish its findings online.",
    "If costs rise above the agreed limit, the council must return to the public for approval.",
    "The first lines are expected to open in time for the university term in two years.",
])

PROMPTS = [
    "The weather today is",
    "In recent economic news,",
    "Scientists have discovered that",
    "The history of Rome shows",
    "Cooking a good meal requires",
]


def run_logic_tests():
    section("PART A: LOGIC TESTS")
    print("  Loading GPT-2 and the paraphraser (about a minute)...")
    tokenizer, model = load_model("gpt2")
    para_tok, para_model = load_paraphraser()

    # ---- Fix 1: very short text ----
    print("\n  Fix 1: very short text")
    for label, text in [("single word", "Hello"), ("empty string", "")]:
        try:
            result = detect_watermark(tokenizer, text)
            check(f"detector handles {label} without crashing", "error" in result)
        except Exception as e:
            check(f"detector handles {label} without crashing", False, f"{type(e).__name__}: {e}")

    # ---- Fix 12: evidence levels ----
    print("\n  Fix 12: evidence levels")
    got = [evidence_strength(z) for z in (-1, 1.9, 2, 3.9, 4, 5.9, 6, 8)]
    want = ["none", "none", "weak", "weak", "moderate", "moderate", "moderate", "strong"]
    check("evidence_strength bands are correct", got == want, str(got))

    # ---- Fix 9: long prompts ----
    print("\n  Fix 9: prompts longer than GPT-2's 1024-token window")
    long_prompt = "word " * 2000
    n_tokens = len(tokenizer.encode(long_prompt))
    for label, fn in [
        ("watermarked generation", lambda: generate_watermarked_text(tokenizer, model, long_prompt, max_new_tokens=10)),
        ("baseline generation", lambda: generate_baseline_text(tokenizer, model, long_prompt, max_new_tokens=10)),
        ("empty prompt", lambda: generate_watermarked_text(tokenizer, model, "", max_new_tokens=5)),
    ]:
        try:
            out = fn()
            check(f"{label} does not crash ({n_tokens}-token prompt)" if "long" in label or "generation" in label
                  else f"{label} does not crash", isinstance(out, str) and len(out) > 0)
        except Exception as e:
            check(f"{label} does not crash", False, f"{type(e).__name__}: {e}")

    # ---- the watermark itself still works ----
    print("\n  Watermark still detectable, baseline still clean")
    detected = 0
    false_pos = 0
    for prompt in PROMPTS:
        wm = generate_watermarked_text(tokenizer, model, prompt, delta=3.0, max_new_tokens=40)
        if detect_watermark(tokenizer, wm)["is_watermarked"]:
            detected += 1
        plain = generate_watermarked_text(tokenizer, model, prompt, delta=0.0, max_new_tokens=40)
        if detect_watermark(tokenizer, plain)["is_watermarked"]:
            false_pos += 1
    check("delta 3.0 text is detected (at least 4 of 5)", detected >= 4, f"{detected}/5")
    check("delta 0 text is not flagged", false_pos == 0, f"{false_pos}/5 false positives")

    # ---- Fix 5: chunked paraphrasing ----
    print("\n  Fix 5: long text is paraphrased in chunks")
    chunks = _split_into_chunks(LONG_TEXT, para_tok)
    sizes = [len(para_tok(c, add_special_tokens=False).input_ids) for c in chunks]
    check("long text splits into several chunks", len(chunks) > 1, f"{len(chunks)} chunks, sizes {sizes}")
    check("every chunk is under about 90 tokens", max(sizes) <= 90)
    check("chunks rejoin to the original text", " ".join(chunks) == LONG_TEXT)
    single = _paraphrase_chunk(LONG_TEXT, para_tok, para_model)
    chunked = remove_watermark_paraphrase(LONG_TEXT, para_tok, para_model)
    single_words, chunked_words = len(single.split()), len(chunked.split())
    check("chunked output covers much more text than one call",
          chunked_words > 1.5 * single_words,
          f"one call {single_words} words, chunked {chunked_words} words, input {len(LONG_TEXT.split())} words")

    # ---- Fix 6: synonym swap quality ----
    print("\n  Fix 6: synonym swap keeps names, numbers and plurals")
    sentence = "The capital of Italy has a long history, and about 10,000 years ago people built a big house there."
    original_words = {w.lower() for w in re.findall(r"[A-Za-z]+", sentence)}
    problems, changed = [], 0
    for seed in range(30):
        out = synonym_substitute(sentence, swap_prob=1.0, seed=seed)
        if out != sentence:
            changed += 1
        for must_keep in ("Italy", "10,000", "years", "has"):
            if must_keep not in out:
                problems.append(f"seed {seed}: lost '{must_keep}' -> {out}")
        if " 's" in out or "_" in out:
            problems.append(f"seed {seed}: bad spacing or underscore -> {out}")
        new_words = [w for w in re.findall(r"[A-Za-z]+", out) if w.lower() not in original_words]
        # "a" -> "an" is the deliberate a/an agreement fix, not a short synonym
        if any(len(w) < 3 and w.lower() not in ("a", "an") for w in new_words):
            problems.append(f"seed {seed}: very short replacement {new_words} -> {out}")
    check("names, numbers, plurals and helper verbs are kept", not problems, problems[0] if problems else "30 seeds")
    check("swaps still happen", changed > 0, f"{changed}/30 seeds changed the sentence")
    print("    example:", synonym_substitute(sentence, swap_prob=0.9, seed=1))

    # ---- the full attack still runs ----
    print("\n  Compound attack end to end")
    wm = generate_watermarked_text(tokenizer, model, "The history of Rome shows", delta=4.0, max_new_tokens=40)
    attacked = compound_attack(wm, para_tok, para_model)
    before = detect_watermark(tokenizer, wm)
    after = detect_watermark(tokenizer, attacked)
    check("compound attack returns text", isinstance(attacked, str) and len(attacked) > 0)
    if "z_score" in before and "z_score" in after:
        print(f"    z before {before['z_score']:.2f}, after {after['z_score']:.2f}")


def run_app_tests():
    section("PART B: THE REAL APP, DRIVEN HEADLESSLY")
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        check("streamlit testing framework available", False, str(e))
        return

    try:
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        at = AppTest.from_file(app_path, default_timeout=240)
        at.run()
        check("app starts without errors", len(at.exception) == 0,
              str([e.value for e in at.exception])[:200])

        # Fix 1 in the UI
        at.text_area(key="w_check").set_value("Hello").run()
        at.button(key="detect_btn").click().run()
        check("single word in Detect shows a warning, not a crash",
              len(at.exception) == 0 and any("Not enough text" in w.value for w in at.warning))

        # Fix 3: result survives a slider move
        sample = " ".join(LONG_TEXT.split(". ")[:4])
        at.text_area(key="w_check").set_value(sample).run()
        at.button(key="detect_btn").click().run()
        shown = any("Detection Result" in m.value for m in at.markdown)
        check("Detect shows a result", len(at.exception) == 0 and shown)
        at.slider(key="w_z").set_value(6.0).run()
        still = any("Detection Result" in m.value for m in at.markdown)
        check("result stays after moving the threshold slider", len(at.exception) == 0 and still)

        # Fix 2: inputs survive a tab switch
        at.text_area(key="w_prompt").set_value("My test prompt").run()
        at.slider(key="w_delta").set_value(3.5).run()
        at.button(key="tab_btn_removal").click().run()
        check("Removal Lab opens without errors", len(at.exception) == 0)
        check("Removal Lab is pre-filled with the detected text",
              at.text_area(key="w_removal").value == sample)

        # Fix 7: threshold is shared
        at.slider(key="w_z_removal").set_value(5.0).run()
        at.button(key="tab_btn_generate").click().run()
        check("prompt survives a tab switch", at.text_area(key="w_prompt").value == "My test prompt")
        check("delta survives a tab switch", at.slider(key="w_delta").value == 3.5)
        check("text to check survives a tab switch", at.text_area(key="w_check").value == sample)
        check("threshold set in Removal Lab carries over", at.slider(key="w_z").value == 5.0)
        check("Detect result survives a tab switch",
              any("Detection Result" in m.value for m in at.markdown))
    except Exception as e:
        check("app test harness ran", False, f"{type(e).__name__}: {e}")


def main():
    run_logic_tests()
    run_app_tests()

    section("SUMMARY")
    failed = [name for name, ok in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print(f"  FAILED: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()