"""Zero-dependency checks for the two-fragment trial-fit and seam scan.

Run with: python3 tests/test_trial_fit.py
Uses only the standard library plus the app's own runtime dependencies.
"""
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from frescobench.fitting import (
    apply_pose,
    choose_base_map,
    polygons_overlap,
    resample_polyline,
    rigid_placement,
)
from frescobench.fits import create_seam_scan, parse_fit_payload, station_lookup
from frescobench.store import Store
from frescobench.trial_pair import GAP_MM, _marks, _marks_b, load_trial_pair


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"ok - {message}")


def test_rigid_placement_preserves_distances_and_orientation():
    # A anchors run horizontally; B anchors run vertically.  The rigid motion
    # must rotate B by 90 degrees and translate it onto A without scaling.
    rotation, translation = rigid_placement([[0.0, 0.0], [10.0, 0.0]],
                                            [[2.0, 3.0], [2.0, 13.0]])
    check(math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9),
          "rigid transform keeps a positive unit determinant")
    mapped = apply_pose(np.asarray([[2.0, 3.0], [2.0, 13.0]]), rotation, translation)
    target = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    check(np.allclose(sorted(mapped.tolist()), sorted(target.tolist()), atol=1e-8),
          "rigid transform maps the B anchor set exactly onto the A anchor set")
    # Lengths are preserved for an arbitrary off-anchor point pair.
    pair = np.asarray([[2.0, 3.0], [2.0, 9.0]])
    original = float(np.linalg.norm(pair[1] - pair[0]))
    moved = apply_pose(pair, rotation, translation)
    check(math.isclose(float(np.linalg.norm(moved[1] - moved[0])), original, rel_tol=1e-9),
          "physical distances are preserved by the rigid motion")


def test_boundary_contact_is_not_an_outline_overlap():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    touching = np.array([[10, 0], [20, 0], [20, 10], [10, 10]], dtype=float)
    crossing = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=float)
    check(polygons_overlap(a, touching) is False, "touching fracture faces are not an overlap")
    check(polygons_overlap(a, crossing) is True, "interior-crossing outlines are an overlap")


def test_resample_keeps_polyline_length():
    chain = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 7.0]])
    dense = resample_polyline(chain, 0.5)
    length = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    check(math.isclose(length, 17.0, rel_tol=0.02), "resampling preserves polyline arc length")


def test_demo_fit_adopts_only_passing_candidate(store):
    _, _, fit = load_trial_pair(store)
    by_id = {item["id"]: item for item in fit["candidates"]}
    check(by_id["close"]["passes"] is True, "closed candidate clears both thresholds")
    check(by_id["shifted"]["passes"] is False, "shifted candidate fails the colour-line threshold")
    check(by_id["close"]["color_line_offset_max_mm"] < fit["thresholds"]["maximum_line_offset_mm"],
          "closed candidate colour line is within the offset limit")
    check(by_id["shifted"]["color_line_offset_max_mm"] > fit["thresholds"]["maximum_line_offset_mm"],
          "shifted candidate colour line misses beyond the offset limit")
    check(fit["adoption"]["candidate_id"] == "close", "only the passing candidate is adopted")
    check(fit["base_map_digest"] is not None, "an adopted fit stores a single base-map image")
    return fit


def test_threshold_gate_rejects_when_match_too_short(store):
    tile_a, tile_b, _ = load_trial_pair(store)
    payload = demo_payload(tile_a, tile_b)
    payload["thresholds"]["minimum_matched_mm"] = 10_000.0
    parsed = parse_fit_payload(payload)
    from frescobench.fits import build_fit_record
    record, _, decision = build_fit_record(store, parsed)
    check(decision is None, "no base map is adopted when the match-length gate fails")
    check(all(not item["passes"] for item in record["candidates"]), "every candidate fails the raised gate")
    check(record["base_map_digest"] is None, "no base-map image is stored without an adoption")


def test_ranking_prefers_solid_contact_then_line_offset():
    good = {"id": "good", "passes": True, "solid_contact_mm": 30.0,
            "color_line_offset_max_mm": 2.0, "color_line_offset_mean_mm": 1.0}
    short = {"id": "short", "passes": True, "solid_contact_mm": 12.0,
             "color_line_offset_max_mm": 0.4, "color_line_offset_mean_mm": 0.4}
    decision = choose_base_map([short, good])
    check(decision["candidate_id"] == "good", "longer contacting solid edge wins first")
    check(decision["rule"] == "longest-solid-contact", "the solid-contact rule is recorded")
    tied_one = {"id": "t1", "passes": True, "solid_contact_mm": 20.0,
                "color_line_offset_max_mm": 1.5, "color_line_offset_mean_mm": 1.0}
    tied_two = {"id": "t2", "passes": True, "solid_contact_mm": 20.0,
                "color_line_offset_max_mm": 0.6, "color_line_offset_mean_mm": 0.5}
    decision = choose_base_map([tied_one, tied_two])
    check(decision["candidate_id"] == "t2", "a solid-contact tie is broken by colour-line offset")
    check(decision["rule"] == "solid-contact-tie-broken-by-colour-line-offset",
          "the documented tie-break rule is recorded")


def test_seam_scan_blank_stations_and_provenance(store):
    _, _, fit = load_trial_pair(store)
    scan = create_seam_scan(store, fit["id"], {
        "title": "cross", "start": [0.05, 0.5], "end": [0.95, 0.5],
        "samples": 100, "band_pixels": 4,
    })
    metrics = scan["metrics"]
    check(bool(metrics["blank_stations"]), "the empty seam produces blank stations")
    for row in metrics["rows"]:
        if row["side"] is None:
            check(row["lightness"] is None and row["samples"] == 0 and row["photo_id"] is None,
                  "blank seam stations carry no colour, sample count or photo")
    sides = [row["side"] for row in metrics["rows"] if row["side"]]
    check(set(sides) == {"A", "B"}, "readable stations are attributed to a real fragment side")
    ordered = sorted(sides, key={"A": 0, "B": 1}.get)
    check(sides == ordered, "station order runs A, blank seam, then B")
    a_station = next(row["station"] for row in metrics["rows"] if row["side"] == "A")
    info = station_lookup(store, fit["id"], scan["id"], a_station)
    check(info["photo_id"] == fit["fragments"]["A"]["photo_id"],
          "a reading points back to the original registered photo of its fragment")
    blank_info = station_lookup(store, fit["id"], scan["id"], metrics["blank_stations"][0])
    check(blank_info["side"] is None, "a blank station reports no owning fragment")
    return fit, scan


def test_persistence_across_reopen(store):
    fit, scan = test_seam_scan_blank_stations_and_provenance(store)
    reopened = Store(str(store.directory))
    fit_again = reopened.fit(fit["id"])
    check(fit_again["adoption"]["candidate_id"] == "close", "the adopted placement survives reopening")
    check(fit_again["fragments"]["A"]["photo_id"] == fit["fragments"]["A"]["photo_id"],
          "the chosen source photo is retained with the fit")
    scans = reopened.list_seam_scans(fit["id"])
    check(any(item["id"] == scan["id"] for item in scans), "the fracture scan is retained with the fit")


def demo_payload(tile_a, tile_b):
    inset = 0.01
    anchor_a_x = 0.977 - inset
    anchor_b_x = 0.023 + inset
    shift = 3.6 / 30.0
    return {
        "title": "trial",
        "fragments": {
            "A": {"tile_id": tile_a["id"], "photo_id": tile_a["plane"]["photo_id"], "marks": _marks()},
            "B": {"tile_id": tile_b["id"], "photo_id": tile_b["plane"]["photo_id"], "marks": _marks_b()},
        },
        "candidates": [
            {"id": "close", "label": "close",
             "anchors_a": [[anchor_a_x, 0.12], [anchor_a_x, 0.86]],
             "anchors_b": [[anchor_b_x, 0.12], [anchor_b_x, 0.86]]},
            {"id": "shifted", "label": "shifted",
             "anchors_a": [[anchor_a_x, 0.12], [anchor_a_x, 0.86]],
             "anchors_b": [[anchor_b_x, 0.12 + shift], [anchor_b_x, 0.86 + shift]]},
        ],
        "thresholds": {
            "minimum_matched_mm": 20.0,
            "maximum_line_offset_mm": 2.5,
            "seam_gap_mm": GAP_MM,
        },
    }


def main():
    with tempfile.TemporaryDirectory() as directory:
        store = Store(str(Path(directory) / "data"))
        test_rigid_placement_preserves_distances_and_orientation()
        test_boundary_contact_is_not_an_outline_overlap()
        test_resample_keeps_polyline_length()
        test_demo_fit_adopts_only_passing_candidate(store)
        test_threshold_gate_rejects_when_match_too_short(store)
        test_ranking_prefers_solid_contact_then_line_offset()
        test_persistence_across_reopen(store)
    print("\nAll trial-fit checks passed.")


if __name__ == "__main__":
    main()
