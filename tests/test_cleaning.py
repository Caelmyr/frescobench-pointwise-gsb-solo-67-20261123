"""End-to-end checks for cleaning fragment studies (run with python -I)."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frescobench.cleaning import (  # noqa: E402
    MAX_CANDIDATE_PAIRS,
    create_cleaning_check,
    create_cleaning_study,
    load_mask,
)
from frescobench.demo import load_training  # noqa: E402
from frescobench.model import InputError  # noqa: E402
from frescobench.store import Store  # noqa: E402


def main():
    data_dir = Path(tempfile.mkdtemp(prefix="frescobench-cleaning-"))
    try:
        store = Store(data_dir)
        tiles, created = load_training(store)
        assert created and len(tiles) == 4, "demo material should seed four fragments"
        roles = {photo["role"] for photo in tiles[0]["photos"]}
        assert {"before", "after"} <= roles, roles
        assert len(tiles[0]["photos"]) <= 10

        def study_region_id(tile):
            return next(region["id"] for region in tile["regions"] if region["name"] == "Cleaning study area")

        region_ids = {tile["id"]: study_region_id(tile) for tile in tiles}

        # Threshold 0/0 selects all four pairs with a non-empty strict region.
        study = create_cleaning_study(store, {
            "title": "Batch cleaning comparison",
            "min_pairs": 1,
            "min_common_area_mm2": 0,
            "region_ids": region_ids,
        })
        assert study["study_type"] == "cleaning"
        assert len(study["pairs"]) == 4, len(study["pairs"])
        assert study["common_pixels"] > 64
        assert study["common_area_mm2"] > 0
        mask = load_mask(store.blobs, study["common_mask_digest"])
        assert mask.any()

        # Every pair was classified on exactly the same frozen pixel set.
        for row in study["rows"]:
            assert row["measured_pixels"] == study["common_pixels"], row
        before = [row["before_bare_percent"] for row in study["rows"]]
        after = [row["after_bare_percent"] for row in study["rows"]]
        assert sum(before) > sum(after), (before, after)

        # Unreachable threshold must be rejected rather than silently relaxed.
        try:
            create_cleaning_study(store, {
                "title": "impossible", "min_pairs": 1,
                "min_common_area_mm2": study["common_area_mm2"] * 100,
                "region_ids": region_ids,
            })
        except InputError:
            pass
        else:
            raise AssertionError("over-large area threshold should fail")

        # Lexicographic rule: asking for fewer pairs must keep all four because
        # pair count is maximized before area.
        same = create_cleaning_study(store, {
            "title": "again", "min_pairs": 2, "min_common_area_mm2": 0,
            "region_ids": region_ids,
        })
        assert len(same["pairs"]) == 4
        assert same["common_mask_digest"] == study["common_mask_digest"]

        # Follow-up re-check inside the adopted region uses the frozen mask.
        member = tiles[0]
        check = create_cleaning_check(store, member["id"], {
            "study_id": study["id"], "region_id": study_region_id(member),
        })
        assert check["kind"] == "cleaning-check"
        assert check["metrics"]["eligible_pixels"] <= study["common_pixels"]
        assert check["metrics"]["eligible_pixels"] == check["metrics"]["before"]["measured"]
        assert check["metrics"]["bare_change_percent"] == round(
            check["metrics"]["after"]["percent"]["bare"]
            - check["metrics"]["before"]["percent"]["bare"], 2
        )

        # A small drawn box partly outside the common region is clipped to it.
        small = create_cleaning_check(store, member["id"], {
            "study_id": study["id"],
            "vertices": [[0.40, 0.40], [0.60, 0.40], [0.60, 0.60], [0.40, 0.60]],
        })
        assert 0 < small["metrics"]["eligible_pixels"] <= study["common_pixels"]
        assert small["metrics"]["eligible_percent_of_box"] <= 100

        # A box entirely outside the observed region cannot be measured.
        try:
            create_cleaning_check(store, member["id"], {
                "study_id": study["id"],
                "vertices": [[0.001, 0.001], [0.004, 0.001], [0.004, 0.004], [0.001, 0.004]],
            })
        except InputError:
            pass
        else:
            raise AssertionError("box outside the adopted region must be rejected")

        # A specimen outside the study cannot borrow it.
        outsider = store.tile(tiles[1]["id"])
        study_two = create_cleaning_study(store, {
            "title": "small", "min_pairs": 1, "min_common_area_mm2": 0,
            "region_ids": {tiles[0]["id"]: study_region_id(tiles[0]), tiles[1]["id"]: study_region_id(tiles[1])},
        })
        assert len(study_two["pairs"]) == 2
        try:
            create_cleaning_check(store, tiles[2]["id"], {
                "study_id": study_two["id"], "region_id": study_region_id(tiles[2]),
            })
        except InputError:
            pass
        else:
            raise AssertionError("non-member specimen must be rejected")

        # Frozen guarantee: adding a new photo does not change the study record.
        before_doc = store.study(study["id"])
        member_live = store.tile(member["id"])
        from frescobench.workspace import import_photo
        from frescobench.demo import render_cleaning_photo, bake_field, SERIES
        from frescobench.photos import png_bytes
        extra = render_cleaning_photo(bake_field(SERIES[0]), "before", SERIES[0])
        # Perturb one pixel so the late capture is a distinct digest.
        pixels = extra.load()
        pixels[10, 10] = tuple(min(255, value + 1) for value in pixels[10, 10])
        import_photo(
            store, member["id"], member_live["version"], png_bytes(extra),
            "extra-before-view.png", "before", "late capture",
        )
        after_doc = store.study(study["id"])
        assert after_doc["pairs"] == before_doc["pairs"]
        assert after_doc["common_mask_digest"] == before_doc["common_mask_digest"]
        assert after_doc["rows"] == before_doc["rows"]

        # Re-check remains reproducible after the late capture (cache hit).
        repeated = create_cleaning_check(store, member["id"], {
            "study_id": study["id"], "region_id": study_region_id(member),
        })
        assert repeated["id"] == check["id"]
        assert MAX_CANDIDATE_PAIRS >= 4
        print("cleaning study integration checks passed")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
