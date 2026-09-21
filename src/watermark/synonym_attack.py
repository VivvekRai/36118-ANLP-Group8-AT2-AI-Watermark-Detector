"""
Synonym-substitution attack: swaps individual words for WordNet synonyms.

Weaker on its own than full paraphrasing (it only touches isolated words,
not sentence structure), but useful as a second pass stacked on top of a
paraphrase attack -- stacking attacks compounds the damage to the
watermark's token pattern more than either attack alone.
"""
import random
import re

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

for _pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


def _wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def synonym_substitute(text, swap_prob=0.3):

    words = word_tokenize(text)
    tagged = pos_tag(words)

    new_words = []
    for word, tag in tagged:
        wn_pos = _wordnet_pos(tag)
        if wn_pos and word.isalpha() and random.random() < swap_prob:
            synsets = wordnet.synsets(word, pos=wn_pos)
            candidates = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != word.lower():
                        candidates.add(candidate)
            if candidates:
                replacement = random.choice(list(candidates))
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words.append(replacement)
                continue
        new_words.append(word)

    text_out = " ".join(new_words)
    text_out = re.sub(r"\s+([.,!?;:])", r"\1", text_out)
    return text_out