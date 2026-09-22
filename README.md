# AI Text Watermark Detector

We built this for our 36118 Applied Natural Language Processing as a working implementation of AI text watermarking, the same core idea behind Google's SynthID, which OpenAI, Nvidia, and a few other big names started adopting in 2026.

The short version: you can secretly nudge a language model to favor certain words while it writes, and then prove later, with real statistics, that the nudging happened. No access to the model needed at detection time, just the text itself.

Our original goal was actually to detect watermarks in models like ChatGPT or Claude. That turned out to be technically impossible, detection only works if you know the exact secret scheme used at generation time, and no AI company shares that. So we pivoted to implementing a real, published watermarking method ourselves, end to end: generation, detection, and genuine adversarial attacks against it.

## What is actually in here

- **A working generator** that biases GPT-2's word choices toward a secretly-seeded "green list" while it writes
- **A statistical detector** that re-derives that same green list and runs a z-test to tell you, with a confidence number, whether a piece of text was watermarked
- **A visual breakdown** that colors every word green or red, so you can actually *see* the pattern instead of just trusting a number
- **A Removal Lab** where we try to break our own watermark with a real compound attack: a paraphrasing model rewrites the sentence, then a synonym-substitution pass further disrupts individual word choices
- **A semantic similarity check** that scores how much of the original meaning survived the attack, so "the watermark was defeated" doesn't just mean "we turned the text into gibberish"

That last combination turned out to be the most interesting part of the project. The watermark holds up fine against light edits, but a genuine paraphrase attack knocks it down hard, while still preserving most of the original meaning, and that is not a bug in what we built, it is a documented, expected weakness of this entire class of watermarking. Google's own SynthID team openly admits to the same limitation.

## Try it

The live version is here: https://ai-watermark-detector.streamlit.app/

Or run it yourself:

```bash
git clone https://github.com/VivvekRai/36118-ANLP-Group8-AT2-AI-Watermark-Detector.git
cd 36118-ANLP-Group8-AT2-AI-Watermark-Detector
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
streamlit run app.py
```

First run downloads a few models. Give it a minute or two the first time you try each tab:

| Model | What it's for | Size |
|---|---|---|
| GPT-2 | Writes the watermarked text | ~550MB |
| `humarin/chatgpt_paraphraser_on_T5_base` | Attacks the watermark by rewriting sentences | ~850MB |
| `all-MiniLM-L6-v2` | Scores how much meaning survived an attack | ~80MB |

Everything runs locally on CPU, no API keys, no cloud calls. That is actually a requirement, not a choice, watermarking needs direct access to the model's internal word probabilities, which a hosted API like ChatGPT's never exposes.

## How it actually works

At every word GPT-2 generates, we look at the word right before it, run it through a hash with a secret number, and use that to shuffle the entire ~50,000-word vocabulary into a "green" pile (25%) and a "red" pile (75%). Then we nudge the model to lean toward green.

Later, to check a piece of text, we just redo that same shuffle at every position and count how often the actual words landed in green. Pure chance gives you about 1 in 4. A watermarked sample usually lands closer to 3 in 4. We turn that gap into a z-score, and if it clears a threshold (z > 4 by default), we call it watermarked.

This is a real, published method: Kirchenbauer, Geiping, Wen, Katz, Miers & Goldstein, *"A Watermark for Large Language Models,"* ICML 2023. Our parameters match their recommended defaults exactly, γ = 0.25, δ = 2.0, z-threshold = 4, and our detection formula is the same z-test for proportions they propose. We just implemented it on GPT-2 instead of their OPT-6.7B, and then went further by stress-testing it with our own compound attack and semantic similarity scoring, something their original paper did not cover.

## What it cannot do

Our detector only recognizes text watermarked with *our own* secret key. Paste in something from ChatGPT or Claude and it will correctly say "not watermarked," not because it failed, but because those systems, if they watermark at all, use their own private scheme. It is less "universal AI detector" (like GPTZero, which guesses based on writing style) and more "can I prove this specific system wrote this" (a provable statistical fact, not a guess).

## What we found

- Watermarked text: z-score usually lands between 5 and 10, depending on how hard we push the bias (δ)
- Plain, non-watermarked text: z-score sits close to 0, and we never got a false positive across every threshold we tested
- A real compound attack (paraphrase + synonym-swap) dropped z-scores by roughly 3.6-4 points on average, often enough to flip the verdict entirely
- Those same attacks typically preserved 65-80% of the original meaning, a real, imperfect trade-off between breaking the watermark and keeping the text useful

## Project structure

- src/watermark/
- model_loader.py loads GPT-2
- green_list.py the core green/red vocabulary split
- generator.py watermarked + plain text generation
- detector.py the z-score statistical test
- visualization.py colors tokens green/red for display
- removal.py paraphrase attack + the compound attack
- synonym_attack.py WordNet-based synonym substitution
- similarity.py semantic similarity scoring between original and attacked text
- robustness.py batch evaluation harness (used in testing, not the live UI)
- app.py the Streamlit interface
- tests/hard_test.py batch testing across many prompts and attack types

## Team

Group 8, 36118 Applied NLP, TD School, UTS.

## References

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). *A Watermark for Large Language Models.* Proceedings of the 40th International Conference on Machine Learning, PMLR 202:17061-17084.

Kirchenbauer, J., Geiping, J., Wen, Y., Shu, M., Saifullah, K., Kong, K., Fernando, K., Saha, A., Goldblum, M., & Goldstein, T. (2023). *On the Reliability of Watermarks for Large Language Models.*