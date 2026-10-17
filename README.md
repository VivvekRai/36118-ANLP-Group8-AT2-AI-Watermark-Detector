<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# AquaMark: AI Text Watermark Detector

We built this for 36118 Applied Natural Language Processing as a working implementation of AI text watermarking, the same core idea behind systems such as Google's SynthID-Text.

The short version: you can secretly nudge a language model to favour certain words while it writes, and then show later, with a statistical test, that the nudging happened. No access to the model is needed at detection time, only the text and the secret key.

Our original goal was to detect watermarks in commercial models such as ChatGPT or Claude. We could not validate that, because detection only works if you know the exact scheme and secret key used at generation time, and providers do not share these. So we implemented a published watermarking method ourselves, end to end: generation, detection, and attacks against it.

## What is in here

- A generator that biases GPT-2's word choices toward a secretly seeded "green list" while it writes
- A statistical detector that re-derives the same green list and runs a z-test to report how strong the evidence is that a piece of text was watermarked
- A visual breakdown that colours every token green or red, so you can see the pattern instead of just trusting a number
- A Removal Lab where we try to break our own watermark: a paraphrasing model rewrites the text, then a synonym-substitution pass further disrupts individual word choices
- A semantic similarity check that scores how much of the original meaning survived the attack, so "the watermark was defeated" does not just mean "we turned the text into gibberish"

The most interesting result is how little editing it takes. In our final runs (three seeds, 15 watermarked texts that were detected before any attack), one pass through the paraphrasing model removed detection from 14 of 15 texts, and the compound attack from all 15, while mean similarity stayed near 0.8. Random deletion of 15% of the words removed it from only 3 of 15. Adding synonyms to the same paraphrase gave no further benefit. This is a known weakness of green-list watermarking in general and is discussed in the literature (see References), not a bug specific to our build.

## Try it

The live version is here: https://ai-watermark-detector.streamlit.app/

Or run it yourself:

```
git clone https://github.com/VivvekRai/36118-ANLP-Group8-AT2-AI-Watermark-Detector.git
cd 36118-ANLP-Group8-AT2-AI-Watermark-Detector
python -m venv venv
venv\Scripts\activate            # Windows (cmd). On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads a few models. Give it a minute or two the first time you use each tab.

| Model | What it is for | Size |
|---|---|---|
| GPT-2 | Writes the watermarked text | ~550MB |
| humarin/chatgpt_paraphraser_on_T5_base | Attacks the watermark by rewriting sentences | ~850MB |
| all-MiniLM-L6-v2 | Scores how much meaning survived an attack | ~80MB |

Everything runs on CPU with no API keys. Watermarking needs direct access to the model's next-token probabilities, which hosted chat APIs generally do not expose.

Package versions are pinned in `requirements.txt` (tested on Python 3.14).

## Reproduce the results

```
python tests/hard_test.py --seed 1
python tests/hard_test.py --seed 2
python tests/hard_test.py --seed 3
python tests/summarise_runs.py > tests/summary.txt
```

Each run takes about 7 to 9 minutes on a CPU. Results are repeatable on the same machine and package versions, but we have not tested other machines.

## How it works

At every step GPT-2 looks at the previous token, hashes it together with a secret key, and uses the result to split the whole vocabulary (about 50,000 tokens) into a "green" list (25%) and a "red" list (75%). A bias δ is added to the green tokens' scores before the next token is chosen, so green tokens become more likely.

To check a piece of text, we redo the same split at every position and count how often the actual tokens fall in the green list. Without a watermark, about 1 in 4 tokens is green by chance. Watermarked text has a higher green fraction, and how much higher depends on δ and on how predictable the text is. We turn the gap into a z-score with a one-proportion z-test. If z is above the threshold (4.0 by default), we flag the text as watermarked.

The app also reports an evidence level based on z: none (below 2), weak (2 to under 4), moderate (4 to 6), strong (above 6). This is an indication of how unlikely the result is by chance, not a probability that the text is watermarked.

This follows Kirchenbauer, Geiping, Wen, Katz, Miers & Goldstein, "A Watermark for Large Language Models" (ICML 2023). We use their soft green-list scheme with γ = 0.25, a configurable δ (the app lets you set it), and the same z-test for proportions, with a default threshold of 4. Their main experiments used OPT-1.3B; we use GPT-2 (124M) so it runs on a laptop CPU. Their paper also evaluates attacks on the watermark; our contribution is a hands-on lab that combines paraphrasing with synonym substitution and measures meaning preserved with a sentence-embedding similarity score.

## What it cannot do

The detector only recognises text watermarked with our own secret key. Text from ChatGPT or Claude will correctly come back as "no evidence of our watermark". That says nothing about whether those systems watermark their output, because any scheme they use is private. This is not a general "AI text detector" (such tools guess from writing style). It answers a narrower question: does this text carry our watermark?

Other limits:
- Very short text gives unreliable statistics. The app warns below 30 tokens.
- GPT-2 has a 1024-token context. Longer prompts are trimmed from the left.
- The synonym attack uses WordNet and can produce awkward word choices.

## What we found

These results come from three seeded runs of tests/hard_test.py (seeds 1 to 3) on commit 1917d9f. The raw texts and scores are saved in tests/results_*_seed*.json, and tests/summary.txt pools them. The sample is small, so treat the results as preliminary.

- With no watermark (δ = 0), none of 30 GPT-2 texts was flagged. These are model texts, not human writing, so this is not a false-positive rate for human text.
- At δ = 2.0, 12 of 15 texts were detected. At δ = 3.0 and 4.0, all 15 were. At δ = 0.5, none were.
- Of the 15 detected texts, paraphrasing defeated 14, the compound attack 15, and random deletion 3.
- Rewrites that removed the watermark kept a mean MiniLM similarity of about 0.80. Similarity is a rough proxy, not a measure of meaning.
- Applying the synonym step to the same paraphrase changed the mean z by -0.08 and lowered similarity by about 0.04, so it added nothing measurable.

## Project structure

```
app.py                          the Streamlit interface
src/watermark/
    model_loader.py             loads GPT-2
    green_list.py               the green/red vocabulary split
    generator.py                watermarked and plain text generation
    detector.py                 the z-score test and evidence levels
    visualization.py            colours tokens green/red for display
    removal.py                  paraphrase attack and the compound attack
    synonym_attack.py           WordNet-based synonym substitution
    similarity.py               semantic similarity between original and attacked text
    robustness.py               batch evaluation harness (testing, not the live UI)
tests/
    hard_test.py                batch tests with fixed seeds (python tests/hard_test.py --seed 1)
    summarise_runs.py           pools the saved runs into tests/summary.txt
    make_fig4.py                draws the Figure 4 scatter plot from the saved runs
    smoke_test.py               quick checks of the logic and the app (python tests/smoke_test.py)
    results_*_seed*.txt/.json   saved results of the three final runs
    summary.txt                 pooled results
docs/                           figures
```

## Team

Team Aqua, Group 8, 36118 Applied NLP, TD School, UTS.

## References

Dathathri, S., See, A., Ghaisas, S., et al. (2024). Scalable watermarking for identifying large language model outputs. *Nature, 634*, 818-823.

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). A watermark for large language models. *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 17061-17084.

Kirchenbauer, J., Geiping, J., Wen, Y., Shu, M., Saifullah, K., Kong, K., Fernando, K., Saha, A., Goldblum, M., & Goldstein, T. (2024). On the reliability of watermarks for large language models. *International Conference on Learning Representations (ICLR 2024)*.

Krishna, K., Song, Y., Karpinska, M., Wieting, J., & Iyyer, M. (2023). Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. *Advances in Neural Information Processing Systems, 36*.