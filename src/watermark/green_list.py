"""
Core green/red-list mechanism.

At every generation step, the previous token (combined with a secret key)
seeds a deterministic pseudorandom split of the vocabulary into a "green"
list and a "red" list. Because the split is reproducible given the same
previous token and secret key, it can be recomputed later by a detector
with no access to the generation process itself.
"""
import torch


def get_green_list(
    prev_token_id: int,
    vocab_size: int,
    green_ratio: float = 0.25,
    secret_key: int = 15485863,
) -> set:
    """
    Deterministically compute the green-list token IDs for one generation
    step, seeded by the previous token.

    Parameters
    ----------
    prev_token_id : the token immediately before the position being generated
    vocab_size : size of the tokenizer's vocabulary
    green_ratio : fraction of the vocabulary considered "green" (default 25%)
    secret_key : shared secret between generator and detector

    Returns
    -------
    set of token IDs belonging to the green list at this step
    """
    seed = (prev_token_id * secret_key) % (2**32)

    rng = torch.Generator()
    rng.manual_seed(seed)

    green_list_size = int(vocab_size * green_ratio)
    permuted = torch.randperm(vocab_size, generator=rng)
    green_list = permuted[:green_list_size]

    return set(green_list.tolist())