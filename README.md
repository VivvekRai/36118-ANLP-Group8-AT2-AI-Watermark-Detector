<p align="center">
  <img src="docs/ccds_logo.png" alt="CCDS logo" width="260">
</p>

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

The most interesting result is the last combination. In our tests the watermark held up against light edits, but a paraphrase attack lowered the z-score sharply while keeping most of the original meaning. This is a known weakness of green-list watermarking in general and is discussed in the literature (see References), not a bug specific to our build.

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

These are results from our own runs. Exact numbers vary with prompt, δ, text length and random seed. See `tests/hard_test.py` to reproduce.

- Watermarked text at moderate-to-high δ usually gave z-scores well above the threshold of 4
- Plain text without a watermark gave z-scores near 0
- A paraphrase plus synonym-swap attack lowered z-scores substantially and often flipped the verdict
- The attacked text kept a fair amount of the original meaning, according to the similarity score. This is a trade-off between breaking the watermark and keeping the text useful

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
    hard_test.py                batch testing across many prompts and attack types
    smoke_test.py               quick checks of the logic and the app (python tests/smoke_test.py)
```

## Team

Group 8, 36118 Applied NLP, TD School, UTS.

## References

Dathathri, S., See, A., Ghaisas, S., et al. (2024). Scalable watermarking for identifying large language model outputs. *Nature, 634*, 818-823.

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). A watermark for large language models. *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 17061-17084.

Kirchenbauer, J., Geiping, J., Wen, Y., Shu, M., Saifullah, K., Kong, K., Fernando, K., Saha, A., Goldblum, M., & Goldstein, T. (2024). On the reliability of watermarks for large language models. *International Conference on Learning Representations (ICLR 2024)*.

Krishna, K., Song, Y., Karpinska, M., Wieting, J., & Iyyer, M. (2023). Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. *Advances in Neural Information Processing Systems, 36*.