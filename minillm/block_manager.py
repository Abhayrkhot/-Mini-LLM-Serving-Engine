"""Paged KV-cache allocator with reference counting and prefix caching.

The KV cache is a pool of fixed-size blocks. Each sequence owns an ordered
``block_table`` mapping its logical token positions to physical blocks, so memory
is allocated one block at a time as the sequence grows instead of reserving
``max_model_len`` up front.

Prefix caching: once a block is full and its K/V has been computed, it is
registered under a hash chained from all preceding blocks (so a hit on block i
implies the whole prefix up to i matches). A later request whose prompt starts
with the same tokens reuses those blocks and skips recomputing them. Blocks are
reference counted; a block whose count drops to zero goes back on the free list
but keeps its hash, so it can still be revived until something else needs the
memory. The free list is LRU: blocks with no cached content are handed out
first, then the least recently freed cached blocks.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence as SequenceABC

from minillm.sequence import Sequence
from minillm.utils import cdiv


class Block:
    __slots__ = ("block_id", "hash", "ref_count", "token_ids")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None
        self.token_ids: tuple[int, ...] | None = None


def chain_hash(parent: int | None, token_ids: tuple[int, ...]) -> int:
    # Python's tuple-of-int hash is deterministic across processes (only str and
    # bytes hashing is randomized). Hits are also verified against the stored
    # token ids, so a collision needs both a 64-bit hash collision and identical
    # block contents with a different prefix.
    return hash((parent, token_ids))


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
        watermark: float = 0.01,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        # Eviction order: popitem(last=False) takes the front.
        self._free: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks))
        self._cached: dict[int, int] = {}  # hash -> block id
        # Blocks kept back when admitting new sequences, so running sequences
        # have room to grow without immediately preempting each other.
        self.watermark_blocks = int(watermark * num_blocks)

        # Cumulative prefix-cache stats, in tokens.
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def usage(self) -> float:
        return 1.0 - len(self._free) / len(self.blocks)

    def find_cached_prefix(self, token_ids: SequenceABC[int]) -> list[int]:
        """Return the longest run of cached blocks matching the start of ``token_ids``.

        Never matches the block holding the last token: the model has to run on
        at least one token to produce logits, and that token's K/V must be
        written into a block this sequence owns.
        """
        if not self.enable_prefix_caching:
            return []
        bs = self.block_size
        matched: list[int] = []
        parent: int | None = None
        for i in range((len(token_ids) - 1) // bs):
            toks = tuple(token_ids[i * bs : (i + 1) * bs])
            h = chain_hash(parent, toks)
            block_id = self._cached.get(h)
            if block_id is None or self.blocks[block_id].token_ids != toks:
                break
            matched.append(block_id)
            parent = h
        return matched

    def _num_new_blocks(self, seq: Sequence, num_new_tokens: int, num_prefix_blocks: int) -> int:
        computed = seq.num_computed_tokens + num_prefix_blocks * self.block_size
        have = len(seq.block_table) + num_prefix_blocks
        return max(0, cdiv(computed + num_new_tokens, self.block_size) - have)

    def can_allocate(
        self,
        seq: Sequence,
        num_new_tokens: int,
        prefix: SequenceABC[int] = (),
        *,
        admit: bool = False,
    ) -> bool:
        need = self._num_new_blocks(seq, num_new_tokens, len(prefix))
        # Reviving a cached block that sits on the free list consumes a free slot.
        revive = sum(1 for b in prefix if self.blocks[b].ref_count == 0)
        reserve = self.watermark_blocks if admit else 0
        return need + revive + reserve <= len(self._free)

    def allocate(self, seq: Sequence, num_new_tokens: int, prefix: SequenceABC[int] = ()) -> None:
        """Give ``seq`` enough blocks to hold ``num_new_tokens`` more computed tokens."""
        fresh = not seq.block_table
        if prefix and not (fresh and seq.num_computed_tokens == 0):
            raise ValueError("a cached prefix can only be attached to a fresh sequence")
        if not self.can_allocate(seq, num_new_tokens, prefix):
            raise RuntimeError("out of KV-cache blocks; call can_allocate first")

        if fresh and self.enable_prefix_caching:
            self.prefix_query_tokens += seq.num_tokens
            self.prefix_hit_tokens += len(prefix) * self.block_size

        for block_id in prefix:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                del self._free[block_id]
            block.ref_count += 1
            seq.block_table.append(block_id)
            assert block.hash is not None
            seq.block_hashes.append(block.hash)
        seq.num_computed_tokens += len(prefix) * self.block_size
        if fresh:
            seq.num_cached_tokens = len(prefix) * self.block_size

        for _ in range(self._num_new_blocks(seq, num_new_tokens, 0)):
            seq.block_table.append(self._take_free_block())

    def try_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        """``allocate`` for a sequence already holding blocks, if the blocks are free.

        The scheduler's hot path: a decoding sequence needs a new block only once
        every ``block_size`` steps, so most calls return without touching the pool.
        """
        bs = self.block_size
        need = (seq.num_computed_tokens + num_new_tokens + bs - 1) // bs - len(seq.block_table)
        if need <= 0:
            return True
        if need > len(self._free):
            return False
        for _ in range(need):
            seq.block_table.append(self._take_free_block())
        return True

    def _take_free_block(self) -> int:
        block_id, _ = self._free.popitem(last=False)
        block = self.blocks[block_id]
        if block.hash is not None:
            if self._cached.get(block.hash) == block_id:
                del self._cached[block.hash]
            block.hash = None
            block.token_ids = None
        block.ref_count = 1
        return block_id

    def free(self, seq: Sequence) -> None:
        """Release every block ``seq`` holds and reset its cache state."""
        # Release the tail first so it lands nearer the front of the eviction
        # order: a prompt's last blocks are the least likely to be shared.
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise AssertionError(f"block {block_id} freed twice")
            if block.ref_count == 0:
                self._free[block_id] = None
                if block.hash is None:
                    self._free.move_to_end(block_id, last=False)
        seq.block_table.clear()
        seq.block_hashes.clear()
        seq.num_computed_tokens = 0

    def reset_prefix_cache(self) -> None:
        """Forget every cached block that no running sequence holds (benchmarks, admin)."""
        for block_id in list(self._free):
            block = self.blocks[block_id]
            if block.hash is not None:
                self._cached.pop(block.hash, None)
                block.hash = None
                block.token_ids = None

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Register blocks of ``seq`` that became full of computed K/V this step."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        # Computed tokens can run ahead of the known ones: with async scheduling a
        # step in flight may be computing a token the CPU has not seen yet.
        num_full = min(seq.num_computed_tokens, len(seq.token_ids)) // bs
        for i in range(len(seq.block_hashes), num_full):
            toks = tuple(seq.token_ids[i * bs : (i + 1) * bs])
            parent = seq.block_hashes[i - 1] if i > 0 else None
            h = chain_hash(parent, toks)
            seq.block_hashes.append(h)
            block = self.blocks[seq.block_table[i]]
            if block.hash is None and h not in self._cached:
                block.hash = h
                block.token_ids = toks
                self._cached[h] = block.block_id
