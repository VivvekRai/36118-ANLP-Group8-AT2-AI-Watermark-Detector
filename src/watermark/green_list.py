import torch


def get_green_list(
    prev_token_id: int,
    vocab_size: int,
    green_ratio: float = 0.25,
    secret_key: int = 15485863,
) -> set:

    seed = (prev_token_id * secret_key) % (2**32)

    rng = torch.Generator()
    rng.manual_seed(seed)

    green_list_size = int(vocab_size * green_ratio)
    permuted = torch.randperm(vocab_size, generator=rng)
    green_list = permuted[:green_list_size]

    return set(green_list.tolist())