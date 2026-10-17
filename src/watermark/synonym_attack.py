import random

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Download NLTK data only if it is missing (not on every import).
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
for _pkg, _path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

# Only base-form tags are swapped, because WordNet gives base forms. Swapping an
# inflected word (years, walked, bigger) for a base form breaks the grammar.
_SWAPPABLE_TAGS = {
    "NN": wordnet.NOUN,
    "JJ": wordnet.ADJ,
    "RB": wordnet.ADV,
    "VB": wordnet.VERB,
    "VBP": wordnet.VERB,
}

# Helper verbs, modals and negations are never swapped.
_LEAVE_ALONE = frozenset({
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "done", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "not", "never", "no", "nor",
    "get", "got", "let", "make", "made", "say", "said",
})

_VOWELS = "aeiou"


def _clean_candidates(word, wn_pos):

    candidates = set()
    for syn in wordnet.synsets(word, pos=wn_pos)[:3]:
        for lemma in syn.lemmas():
            name = lemma.name()
            if not name.isalpha() or len(name) < 3:
                continue  # drops multi-word ("make_up"), hyphenated and abbreviated forms
            if name != name.lower():
                continue  # drops proper nouns such as 'Washington'
            if name.lower() == word.lower():
                continue
            if lemma.count() < 1:
                continue  # drops rare or archaic words
            candidates.add(name)
    return sorted(candidates)  # sorted so a fixed seed gives a repeatable result


def _fix_articles(words, replaced):

    for i in range(len(words) - 1):
        if (i + 1) in replaced and words[i].lower() in ("a", "an"):
            wants_an = words[i + 1][0].lower() in _VOWELS
            article = "an" if wants_an else "a"
            words[i] = article.capitalize() if words[i][0].isupper() else article
    return words


def synonym_substitute(text, swap_prob=0.3, seed=None):
    rng = random.Random(seed)

    words = word_tokenize(text)
    tagged = pos_tag(words)

    new_words = []
    replaced = set()
    for index, (word, tag) in enumerate(tagged):
        wn_pos = _SWAPPABLE_TAGS.get(tag)
        swappable = (
            wn_pos is not None
            and word.isalpha()
            and len(word) >= 3
            and not word.isupper()          # acronyms (AI, GPT)
            and word.lower() not in _LEAVE_ALONE
        )
        if swappable and rng.random() < swap_prob:
            candidates = _clean_candidates(word, wn_pos)
            if candidates:
                replacement = rng.choice(candidates)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                else:
                    replacement = replacement.lower()
                new_words.append(replacement)
                replaced.add(index)
                continue
        new_words.append(word)

    new_words = _fix_articles(new_words, replaced)
    return TreebankWordDetokenizer().detokenize(new_words)