"""Block tables on the device, and the per-token / per-page index math done there.

The block manager owns every sequence's block table as a Python list. Building
a step's KV-cache slot for every new token and FlashInfer's page list for
every sequence from those lists costs Python time proportional to the context:
at 256 decoding sequences that is tens of thousands of list entries a step.

Instead each running sequence gets a row in a device tensor that mirrors its
table. A sequence gains a block only every ``block_size`` tokens, so each step
sends just the few new entries; slots and page indices are then a handful of
small tensor ops. The host side does O(sequences) work per step.
"""

from __future__ import annotations

import torch

from minillm.sequence import Sequence


class DeviceBlockTables:
    """``rows`` sequence rows plus one padding row whose pages are all ``pad_block``."""

    def __init__(self, rows: int, max_blocks: int, pad_block: int, device: torch.device) -> None:
        self.table = torch.full((rows + 1, max_blocks), pad_block, dtype=torch.int32, device=device)
        self.pad_row = rows
        self._row: dict[Sequence, int] = {}
        self._owner: list[Sequence | None] = [None] * rows
        self._synced = [0] * rows  # entries of the row that match the sequence's table
        self._version = [-1] * rows  # the sequence's preemption count when last synced
        self._last_used = [-1] * rows
        self._free = list(range(rows - 1, -1, -1))
        self._step = 0

    def begin_step(self) -> None:
        self._step += 1

    def row(self, seq: Sequence, rows: list[int], cols: list[int], vals: list[int]) -> int:
        """The sequence's row, appending to (rows, cols, vals) the entries the device lacks."""
        r = self._row.get(seq)
        if r is None:
            r = self._assign(seq)
        self._last_used[r] = self._step
        if self._version[r] != seq.num_preemptions:
            # Preemption freed the table and readmission rebuilt it: resend all of it.
            self._version[r] = seq.num_preemptions
            self._synced[r] = 0
        table = seq.block_table
        have, n = self._synced[r], len(table)
        if n > have:
            rows.extend([r] * (n - have))
            cols.extend(range(have, n))
            vals.extend(table[have:n])
            self._synced[r] = n
        return r

    def _assign(self, seq: Sequence) -> int:
        if self._free:
            r = self._free.pop()
        else:
            # Every row is taken: reuse the least recently used one that this step
            # has not touched (finished sequences' rows age out this way). A step
            # holds at most max_num_seqs sequences, so one always exists.
            idle = [i for i in range(len(self._owner)) if self._last_used[i] < self._step]
            if not idle:
                raise RuntimeError(
                    f"more than {len(self._owner)} sequences in one step (max_num_seqs)"
                )
            r = min(idle, key=self._last_used.__getitem__)
            old = self._owner[r]
            if old is not None:
                del self._row[old]
        self._row[seq] = r
        self._owner[r] = seq
        self._synced[r] = 0
        self._version[r] = seq.num_preemptions
        return r

    def write(self, rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor) -> None:
        """Apply this step's new entries (device tensors, int64)."""
        if rows.numel():
            self.table[rows, cols] = vals.to(torch.int32)

    def slots(self, rows: torch.Tensor, positions: torch.Tensor, block_size: int) -> torch.Tensor:
        """Flat KV-cache slot of each token, given its sequence's row and its position."""
        blocks = self.table[rows, positions // block_size].long()
        out: torch.Tensor = blocks * block_size + positions % block_size
        return out

    def page_indices(
        self, rows: torch.Tensor, pages: torch.Tensor, page_start: torch.Tensor, total: int
    ) -> torch.Tensor:
        """FlashInfer's flat page list: the first ``pages[i]`` blocks of row ``rows[i]``."""
        seq = torch.repeat_interleave(
            torch.arange(rows.numel(), device=rows.device), pages, output_size=total
        )
        within = torch.arange(total, device=rows.device) - page_start[seq]
        indices: torch.Tensor = self.table[rows[seq], within]
        return indices
