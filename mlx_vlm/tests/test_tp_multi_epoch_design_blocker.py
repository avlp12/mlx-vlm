"""CPU-only failing-contract fixtures for TP row lifecycle protocol design."""

from mlx_vlm.models.cache import KVCache
from mlx_vlm.tp import worker as W


class _LM:
    def make_cache(self):
        return [KVCache()]


def test_current_worker_discards_incumbent_when_donor_prefill_opens_epoch():
    state = W._WorkerState(_LM())
    incumbent = state.new_cache(1, [2, 0])
    donor = state.new_cache(2, [1])
    assert donor is state.caches[2]
    assert 1 not in state.caches
    assert incumbent is not donor


def test_required_multi_epoch_contract_is_currently_absent():
    state = W._WorkerState(_LM())
    state.new_cache(1, [2, 0])
    state.new_cache(2, [1])
    # This is the prerequisite for an exact EXTEND(dest=1, source=2).
    assert set(state.caches) != {1, 2}
