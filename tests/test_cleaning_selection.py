"""Selection-rule unit tests for cleaning studies (run with the repo on PYTHONPATH)."""
import numpy as np

from frescobench.cleaning import CandidatePair, select_pairs
from frescobench.model import InputError


SHAPE = (10, 10)
SCALE = (1.0, 1.0)


def mask(slice_spec):
    result = np.zeros(SHAPE, dtype=bool)
    result[slice_spec] = True
    return result


def build(coverages):
    pairs = []
    masks = []
    for index, coverage in enumerate(coverages):
        identifier = f"{index:024x}"
        pairs.append(CandidatePair(identifier, f"T{index}", "b" + identifier, "a" + identifier, "r", "r", 10.0, 10.0))
        masks.append(coverage)
    canonical = {}
    for pair, coverage in zip(pairs, masks):
        canonical[pair.before_photo_id] = coverage
        canonical[pair.after_photo_id] = coverage
    return pairs, canonical


def test_pairs_outrank_area():
    pairs, canonical = build([
        mask(np.s_[0:10, 0:10]),
        mask(np.s_[0:6, 0:6]),
        mask(np.s_[0:4, 0:4]),
        mask(np.s_[0:2, 0:2]),
    ])
    selected, area = select_pairs(pairs, canonical, SCALE, 1, 0)
    assert len(selected) == 4 and area == 4.0


def test_area_floor_drops_pairs():
    pairs, canonical = build([
        mask(np.s_[0:10, 0:10]),
        mask(np.s_[0:6, 0:6]),
        mask(np.s_[0:4, 0:4]),
        mask(np.s_[0:2, 0:2]),
    ])
    selected, area = select_pairs(pairs, canonical, SCALE, 1, 25)
    assert len(selected) == 2 and area == 36.0
    selected, area = select_pairs(pairs, canonical, SCALE, 1, 10)
    assert len(selected) == 3 and area == 16.0


def test_deterministic_tie_break():
    pairs = [
        CandidatePair("b" * 24, "B", "b1", "a1", "r", "r", 10.0, 10.0),
        CandidatePair("a" * 24, "A", "b2", "a2", "r", "r", 10.0, 10.0),
    ]
    canonical = {}
    for pair, coverage in zip(pairs, [mask(np.s_[0:5, 0:5]), mask(np.s_[5:10, 5:10])]):
        canonical[pair.before_photo_id] = coverage
        canonical[pair.after_photo_id] = coverage
    selected, area = select_pairs(pairs, canonical, SCALE, 1, 10)
    assert len(selected) == 1 and area == 25.0 and selected[0].tile_id == "a" * 24


def test_infeasible_thresholds():
    pairs, canonical = build([
        mask(np.s_[0:10, 0:10]),
        mask(np.s_[0:2, 0:2]),
    ])
    try:
        select_pairs(pairs, canonical, SCALE, 2, 5)
    except InputError:
        return
    raise AssertionError("thresholds above the strict overlap must fail")


if __name__ == "__main__":
    test_pairs_outrank_area()
    test_area_floor_drops_pairs()
    test_deterministic_tie_break()
    test_infeasible_thresholds()
    print("selection rule unit checks passed")
