"""The device mirror of block tables, and the slot / page-index math done on it."""

from __future__ import annotations

import torch

from minillm.executor.block_tables import DeviceBlockTables
from minillm.sampling import SamplingParams
from minillm.sequence import Sequence


def seq(name: str, table: list[int]) -> Sequence:
    s = Sequence(name, [1, 2, 3], SamplingParams(), 0.0)
    s.block_table = list(table)
    return s


def sync(tables: DeviceBlockTables, seqs: list[Sequence]) -> list[int]:
    """One step: sync every sequence's row, then check the device copy."""
    tables.begin_step()
    rows_, cols, vals = [], [], []
    rows = [tables.row(s, rows_, cols, vals) for s in seqs]
    tables.write(torch.tensor(rows_), torch.tensor(cols), torch.tensor(vals))
    for s, r in zip(seqs, rows, strict=True):
        assert tables.table[r, : len(s.block_table)].tolist() == s.block_table
    return rows


def test_rows_mirror_tables_through_growth_reuse_and_preemption() -> None:
    tables = DeviceBlockTables(rows=2, max_blocks=8, pad_block=99, device=torch.device("cpu"))
    a, b, c = seq("a", [1, 2, 3]), seq("b", [4, 5]), seq("c", [6])
    ra, rb = sync(tables, [a, b])
    a.block_table.append(7)  # growth: only the new entry is sent
    rows_, cols, vals = [], [], []
    tables.begin_step()
    tables.row(a, rows_, cols, vals)
    tables.row(b, rows_, cols, vals)
    assert (rows_, cols, vals) == ([ra], [3], [7])
    tables.write(torch.tensor(rows_), torch.tensor(cols), torch.tensor(vals))
    # Both rows are taken and a is done: c gets the least recently used row, a's.
    assert sync(tables, [c, b]) == [ra, rb]
    # Preemption freed b's table; readmission rebuilt it from scratch.
    b.block_table[:] = [8, 9]
    b.num_preemptions += 1
    sync(tables, [c, b])
    assert tables.table[tables.pad_row].tolist() == [99] * 8


def test_slots_and_page_indices_match_python() -> None:
    bs = 4
    tables = DeviceBlockTables(rows=3, max_blocks=6, pad_block=99, device=torch.device("cpu"))
    seqs = [seq("x", [10, 11, 12]), seq("y", [20]), seq("z", [30, 31])]
    rows = sync(tables, seqs)
    positions = [9, 2, 5]
    slots = tables.slots(torch.tensor(rows), torch.tensor(positions), bs)
    expected = [s.block_table[p // bs] * bs + p % bs for s, p in zip(seqs, positions, strict=True)]
    assert slots.tolist() == expected

    pages = [3, 1, 2, 1]  # plus one padding row
    all_rows = [*rows, tables.pad_row]
    starts = [0, 3, 4, 6]
    indices = tables.page_indices(
        torch.tensor(all_rows), torch.tensor(pages), torch.tensor(starts), total=7
    )
    assert indices.tolist() == [10, 11, 12, 20, 30, 31, 99]
