"""N-gram (prompt lookup) drafts."""

from __future__ import annotations

import random

from minillm.ngram import NgramProposer
from minillm.sampling import SamplingParams
from minillm.sequence import Sequence


def seq(tokens: list[int]) -> Sequence:
    return Sequence("r", tokens, SamplingParams(), 0.0)


def brute_force(tokens: list[int], k: int, n_max: int, n_min: int) -> list[int]:
    """The definition: the longest suffix n-gram's most recent earlier occurrence."""
    length = len(tokens)
    for n in range(min(n_max, length - 1), n_min - 1, -1):
        suffix = tokens[length - n :]
        for start in range(length - n - 1, -1, -1):
            if tokens[start : start + n] == suffix:
                return tokens[start + n : start + n + k]
    return []


def test_proposes_what_followed_the_last_occurrence() -> None:
    p = NgramProposer(max_draft=3)
    s = seq([1, 2, 3, 4, 5, 9, 1, 2, 3])
    assert p.propose(s, limit=10) == [4, 5, 9]  # "1 2 3" was followed by 4 5 9
    assert p.propose(s, limit=2) == [4, 5]


def test_prefers_longer_matches_and_the_most_recent_occurrence() -> None:
    p = NgramProposer(max_draft=2)
    # "7" alone last appeared before 8; the 2-gram "6 7" before 5.
    assert p.propose(seq([6, 7, 5, 0, 7, 8, 6, 7]), limit=2) == [5, 0]
    assert p.propose(seq([3, 1, 3, 2, 3]), limit=1) == [2]  # most recent "3" is followed by 2


def test_no_match_and_no_self_match() -> None:
    p = NgramProposer(max_draft=4)
    assert p.propose(seq([1, 2, 3, 4]), limit=4) == []
    assert p.propose(seq([5]), limit=4) == []


def test_index_stays_correct_as_tokens_arrive() -> None:
    rng = random.Random(0)
    for _ in range(30):
        p = NgramProposer(max_draft=4, n_max=3, n_min=1)
        tokens = [rng.randrange(6) for _ in range(5)]
        s = seq(tokens)
        for _ in range(60):
            assert p.propose(s, limit=4) == brute_force(s.token_ids, 4, 3, 1)
            s.token_ids.append(rng.randrange(6))
