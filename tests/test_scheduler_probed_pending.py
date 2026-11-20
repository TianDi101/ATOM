# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PROBED_PENDING: the prefill delayer's pending tokens count a waiting
request past its HBM prefix hit, not at its whole prompt."""

import pytest
from conftest import MockConfig

from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Scheduler


class _Delayer:
    is_local = False
    max_queue_ms = None
    kv_high_watermark = 0.9
    rank_prefill_costs: tuple = ()

    def __init__(self):
        self.seen: list[dict] = []

    def protects_decode(self, running_decode_batch):
        return False

    def should_allow_prefill(self, **kw):
        self.seen.append(kw)
        return PrefillDecision(allow=False, prefillable_ranks=1)

    def notify_prefill_executed(self): ...


def _make(monkeypatch, probed):
    monkeypatch.setenv("ATOM_PROBED_PENDING", str(probed))
    sched = Scheduler(
        MockConfig(max_num_seqs=8, max_num_batched_tokens=64, num_kvcache_blocks=100)
    )
    delayer = _Delayer()
    sched.set_prefill_delayer(delayer)
    return sched, delayer


@pytest.mark.parametrize("probed", [0, 1])
def test_waiting_request_counts_past_its_prefix_hit(monkeypatch, seq_factory, probed):
    sched, delayer = _make(monkeypatch, probed)
    bm = sched.block_manager
    block = bm.hash_block_size
    sched.add(seq_factory(list(range(4 * block))))
    # Three of its four blocks are in this rank's pool.
    monkeypatch.setattr(bm, "can_allocate_memoized", lambda seq: 3)

    sched.schedule()

    assert delayer.seen[-1]["pending_tokens"] == (
        block if probed else min(4 * block, sched.max_num_batched_tokens)
    )


def test_request_the_pool_cannot_take_keeps_the_upper_bound(monkeypatch, seq_factory):
    sched, _ = _make(monkeypatch, 1)
    bm = sched.block_manager
    seq = seq_factory(list(range(2 * bm.hash_block_size)))
    sched.add(seq)
    monkeypatch.setattr(bm, "can_allocate_memoized", lambda seq: -1)

    assert sched._probed_pending_prefill_tokens(seq) == 2 * bm.hash_block_size
