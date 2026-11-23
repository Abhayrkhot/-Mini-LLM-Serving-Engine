from __future__ import annotations

from collections import Counter, defaultdict

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from minillm.block_manager import BlockManager
from minillm.sampling import SamplingParams
from minillm.sequence import Sequence


def make_seq(tokens: list[int], rid: str = "r") -> Sequence:
    return Sequence(rid, tokens, SamplingParams(), 0.0)


def compute(bm: BlockManager, seq: Sequence, n: int) -> None:
    """Allocate for and 'run' n tokens, as one engine step would."""
    bm.allocate(seq, n)
    seq.num_computed_tokens += n
    bm.cache_full_blocks(seq)


def test_allocates_blocks_lazily_and_frees_everything():
    bm = BlockManager(num_blocks=8, block_size=4, watermark=0)
    seq = make_seq(list(range(1, 6)))  # 5 tokens -> 2 blocks
    compute(bm, seq, 5)
    assert len(seq.block_table) == 2
    assert bm.num_free_blocks == 6
    for t in range(3):  # tokens 6, 7, 8 fit in block 2
        seq.token_ids.append(100 + t)
        compute(bm, seq, 1)
    assert len(seq.block_table) == 2
    seq.token_ids.append(200)
    compute(bm, seq, 1)  # token 9 needs block 3
    assert len(seq.block_table) == 3
    bm.free(seq)
    assert bm.num_free_blocks == 8
    assert seq.block_table == [] and seq.num_computed_tokens == 0


def test_prefix_hit_reuses_full_blocks_but_never_the_last_token():
    bm = BlockManager(num_blocks=16, block_size=4, watermark=0)
    first = make_seq(list(range(1, 13)), "a")  # 3 full blocks
    compute(bm, first, 12)

    same = make_seq(list(range(1, 13)), "b")
    hit = bm.find_cached_prefix(same.token_ids)
    assert hit == first.block_table[:2]  # block 3 holds the last token: must be recomputed

    longer = make_seq([*range(1, 13), 99], "c")
    assert bm.find_cached_prefix(longer.token_ids) == first.block_table

    diverged = make_seq([1, 2, 3, 4, 5, 6, 0, 8, 9], "d")
    assert bm.find_cached_prefix(diverged.token_ids) == first.block_table[:1]


def test_shared_blocks_are_reference_counted():
    bm = BlockManager(num_blocks=16, block_size=4, watermark=0)
    a = make_seq(list(range(1, 10)), "a")
    compute(bm, a, 9)
    b = make_seq(list(range(1, 10)), "b")
    hit = bm.find_cached_prefix(b.token_ids)
    bm.allocate(b, b.num_tokens - 8, hit)
    assert b.num_computed_tokens == 8 and b.num_cached_tokens == 8
    assert b.block_table[:2] == a.block_table[:2]
    assert b.block_table[2] != a.block_table[2]
    assert all(bm.blocks[i].ref_count == 2 for i in a.block_table[:2])
    bm.free(a)
    assert all(bm.blocks[i].ref_count == 1 for i in b.block_table[:2])
    assert bm.prefix_hit_tokens == 8 and bm.prefix_query_tokens == 18


def test_freed_cached_blocks_can_be_revived_until_evicted():
    bm = BlockManager(num_blocks=4, block_size=4, watermark=0)
    a = make_seq(list(range(1, 10)), "a")  # 3 blocks, 2 full
    compute(bm, a, 9)
    bm.free(a)
    assert bm.num_free_blocks == 4
    b = make_seq(list(range(1, 10)), "b")
    hit = bm.find_cached_prefix(b.token_ids)
    assert len(hit) == 2
    # Reviving 2 cached blocks + 1 new block uses 3 of the 4 free slots.
    assert bm.can_allocate(b, 1, hit)
    # A different prompt that needs all 4 blocks evicts the cache.
    c = make_seq(list(range(50, 66)), "c")
    compute(bm, c, 16)
    assert bm.find_cached_prefix(b.token_ids) == []


def test_uncached_blocks_are_reused_before_cached_ones():
    bm = BlockManager(num_blocks=6, block_size=4, watermark=0)
    cached = make_seq(list(range(1, 9)), "cached")  # 2 full blocks, both hashed
    compute(bm, cached, 8)
    partial = make_seq([7, 7], "partial")  # 1 partial block, never hashed
    compute(bm, partial, 2)
    cached_ids = list(cached.block_table)
    bm.free(cached)
    bm.free(partial)
    fresh = make_seq(list(range(100, 116)), "fresh")  # needs 4 of the 6 free blocks
    compute(bm, fresh, 16)
    # The 3 never-used blocks and the partial one were consumed; the cache survives.
    assert set(fresh.block_table).isdisjoint(cached_ids)
    assert bm.find_cached_prefix([*range(1, 9), 0]) == cached_ids


def test_watermark_only_applies_to_admission():
    bm = BlockManager(num_blocks=100, block_size=1, watermark=0.1)
    seq = make_seq(list(range(1, 92)))
    assert not bm.can_allocate(seq, 91, admit=True)  # 91 + 10 reserved > 100
    assert bm.can_allocate(seq, 91, admit=False)


def test_rejects_bad_config_and_double_free():
    with pytest.raises(ValueError):
        BlockManager(0, 16)
    bm = BlockManager(4, 4, watermark=0)
    seq = make_seq([1, 2, 3])
    compute(bm, seq, 3)
    table = list(seq.block_table)
    bm.free(seq)
    seq.block_table = table
    with pytest.raises(AssertionError):
        bm.free(seq)


# ---------------------------------------------------------------- property-based

PREFIXES = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 10, 11, 12, 13], [20, 21, 22, 23, 24, 25]]
BS = 4


class BlockManagerMachine(RuleBasedStateMachine):
    """Random admit / decode / free sequences against a model of physical block contents.

    The key safety property: every computed token of every live sequence is
    still physically present in the block its table points to. A refcount bug,
    a premature eviction or a bad prefix hit would let another sequence
    overwrite it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bm = BlockManager(num_blocks=20, block_size=BS, watermark=0)
        self.live: list[Sequence] = []
        self.content: dict[int, dict[int, int]] = defaultdict(dict)
        self.count = 0

    def _compute(self, seq: Sequence, n: int) -> None:
        start = seq.num_computed_tokens
        for pos in range(start, start + n):
            self.content[seq.block_table[pos // BS]][pos % BS] = seq.token_ids[pos]
        seq.num_computed_tokens += n
        self.bm.cache_full_blocks(seq)

    @rule(prefix=st.integers(0, len(PREFIXES) - 1), tail=st.lists(st.integers(1, 40), max_size=9))
    def admit(self, prefix: int, tail: list[int]) -> None:
        seq = make_seq(PREFIXES[prefix] + tail, f"s{self.count}")
        self.count += 1
        hit = self.bm.find_cached_prefix(seq.token_ids)
        n = seq.num_tokens - len(hit) * BS
        if not self.bm.can_allocate(seq, n, hit):
            return
        self.bm.allocate(seq, n, hit)
        self._compute(seq, n)
        self.live.append(seq)

    @precondition(lambda self: self.live)
    @rule(data=st.data(), steps=st.integers(1, 6))
    def decode(self, data: st.DataObject, steps: int) -> None:
        seq = data.draw(st.sampled_from(self.live))
        for _ in range(steps):
            if seq.num_computed_tokens == seq.num_tokens:
                seq.token_ids.append(data.draw(st.integers(1, 40)))
            n = seq.num_tokens - seq.num_computed_tokens
            if not self.bm.can_allocate(seq, n):
                return
            self.bm.allocate(seq, n)
            self._compute(seq, n)

    @precondition(lambda self: self.live)
    @rule(data=st.data())
    def finish(self, data: st.DataObject) -> None:
        seq = data.draw(st.sampled_from(self.live))
        self.bm.free(seq)
        self.live.remove(seq)

    @invariant()
    def refcounts_match_tables(self) -> None:
        held = Counter(b for s in self.live for b in s.block_table)
        for block in self.bm.blocks:
            assert block.ref_count == held.get(block.block_id, 0)
        assert self.bm.num_free_blocks == sum(1 for b in self.bm.blocks if b.ref_count == 0)

    @invariant()
    def computed_tokens_are_intact(self) -> None:
        for s in self.live:
            for pos in range(s.num_computed_tokens):
                assert self.content[s.block_table[pos // BS]][pos % BS] == s.token_ids[pos]

    @invariant()
    def cache_index_is_consistent(self) -> None:
        for h, block_id in self.bm._cached.items():
            assert self.bm.blocks[block_id].hash == h


TestBlockManagerStateful = BlockManagerMachine.TestCase
TestBlockManagerStateful.settings = settings(
    max_examples=300, stateful_step_count=60, deadline=None
)


def test_reset_prefix_cache_forgets_only_unreferenced_blocks():
    bm = BlockManager(num_blocks=16, block_size=4, watermark=0)
    live = make_seq(list(range(1, 10)), "live")
    compute(bm, live, 9)
    done = make_seq(list(range(50, 59)), "done")
    compute(bm, done, 9)
    bm.free(done)
    bm.reset_prefix_cache()
    assert bm.find_cached_prefix([*range(50, 59), 0]) == []
    assert bm.find_cached_prefix([*range(1, 10), 0]) == live.block_table[:2]
