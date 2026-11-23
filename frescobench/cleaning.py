"""Cleaning fragment studies.

A cleaning study compares equally sized wall-painting fragments from one batch
before and after cleaning. Every retained fragment pair is evaluated only on
positions that every selected photograph actually sees: a position missing from
the frame or clipped (over-exposed / black-clipped) in even one selected photo
is excluded for the whole study and can never be back-filled from the other
photographs. Studies are frozen records — later captures or independent
measurements never change the adopted pairs or the adopted common region.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .art import base_rgba, comparison_sheet, coverage_overlay, stored_preview
from .color import corrected_linear
from .coverage import measure_coverage
from .geometry import polygon_mask, region_domain, specimen_frame
from .model import CLEANING_ROLES, InputError, dictionary, flag, identity, integer, scalar, string
from .photos import png_bytes
from .store import key_for
from .study import study_stats


MAX_CANDIDATE_PAIRS = 20
MIN_COMMON_PIXELS = 64
GRID_LIMIT = 1800

# Coverage options reused for every study and every later re-check so the
# frozen classification stays comparable.
STUDY_COVERAGE_OPTIONS = {
    "ambiguity": 0.16,
    "max_residual_fraction": 0.30,
    "minimum_island": 22,
}


@dataclass(frozen=True)
class CandidatePair:
    tile_id: str
    tile_name: str
    before_photo_id: str
    after_photo_id: str
    region_id: str
    region_name: str
    width_mm: float
    height_mm: float


def unique_reference_region(tile, purpose):
    matches = [region for region in tile["regions"] if region["purpose"] == purpose]
    if not matches:
        return None
    if len(matches) > 1:
        raise InputError(
            f"Specimen {tile['name']} has several {purpose} reference patches. "
            "Keep one unambiguous control before building a cleaning study."
        )
    return matches[0]


def pair_diagnostics(store, tile, region_id=None):
    """Return the eligible cleaning pair for a specimen, or a reason why not."""
    if not tile.get("plane"):
        return None, "No registered survey plane.", []
    if tile["width_mm"] is None or tile["height_mm"] is None:
        return None, "Specimen width and height in millimetres are required.", []
    by_role = {}
    for photo in tile["photos"]:
        registered = photo["id"] == tile["plane"]["photo_id"] or photo["id"] in tile["registrations"]
        if photo["role"] in CLEANING_ROLES and registered:
            by_role.setdefault(photo["role"], []).append(photo)
    for role in CLEANING_ROLES:
        if not by_role.get(role):
            stage = "before" if role == "before" else "after"
            return None, f"Missing a registered {stage}-cleaning photo.", []
        if len(by_role[role]) > 1:
            return None, f"More than one {role}-cleaning photo is registered; keep one of each.", []
    if unique_reference_region(tile, "bare") is None:
        return None, "Mark one bare plaster reference patch.", []
    if unique_reference_region(tile, "glazed") is None:
        return None, "Mark one intact pigment reference patch.", []
    inspections = [region for region in tile["regions"] if region["purpose"] == "inspection"]
    region_options = [{"id": region["id"], "name": region["name"]} for region in inspections]
    if region_id:
        region = next((item for item in inspections if item["id"] == region_id), None)
        if region is None:
            return None, "The selected inspection region is no longer on the specimen.", region_options
    else:
        if not inspections:
            return None, "Draw an inspection region for the area under study.", []
        if len(inspections) > 1:
            return None, "Choose which inspection region defines the study area.", region_options
        region = inspections[0]
    pair = CandidatePair(
        tile_id=tile["id"],
        tile_name=tile["name"],
        before_photo_id=by_role["before"][0]["id"],
        after_photo_id=by_role["after"][0]["id"],
        region_id=region["id"],
        region_name=region["name"],
        width_mm=float(tile["width_mm"]),
        height_mm=float(tile["height_mm"]),
    )
    return pair, None, region_options


def resample_mask(mask, target_shape):
    """Resample a boolean plane mask onto another grid, keeping only full support."""
    if mask.shape == target_shape:
        return mask.copy()
    source_y, source_x = np.indices(target_shape, dtype=np.float32)
    coords = (
        (source_y + 0.5) * mask.shape[0] / target_shape[0] - 0.5,
        (source_x + 0.5) * mask.shape[1] / target_shape[1] - 0.5,
    )
    sampled = ndimage.map_coordinates(mask.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    return sampled >= 0.999


def canonical_grid(pairs, frames):
    """One shared grid for all fragments; all fragments share mm dimensions."""
    if len({(pair.width_mm, pair.height_mm) for pair in pairs}) != 1:
        raise InputError("Only equally sized fragments with identical millimetre dimensions can share one region.")
    extents = [frames[key].extent for pair in pairs for key in (pair.before_photo_id, pair.after_photo_id)]
    width = max(extent[0] for extent in extents)
    height = max(extent[1] for extent in extents)
    longest = max(width, height)
    if longest > GRID_LIMIT:
        factor = GRID_LIMIT / longest
        width = max(16, int(round(width * factor)))
        height = max(16, int(round(height * factor)))
    shape = (height, width)
    pixel_scale = (pairs[0].width_mm / width, pairs[0].height_mm / height)
    return shape, pixel_scale


def pair_observed_mask(tile, pair, frame):
    """Positions this photo sees inside the chosen inspection area.

    Exclusion regions are removed, out-of-frame positions through
    ``frame.valid`` and clipped (over-exposed / black) positions through
    ``frame.clipped``.
    """
    region = next(item for item in tile["regions"] if item["id"] == pair.region_id)
    domain = region_domain(frame.extent, tile, region)
    return domain & frame.valid & ~frame.clipped


def mask_png(mask):
    return png_bytes(Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L"))


def load_mask(blobs, digest):
    with Image.open(blobs.require(digest)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def mask_to_frame(mask, frame):
    """Map the canonical study mask back onto a registered photo frame."""
    target_shape = frame.valid.shape
    if mask.shape == target_shape:
        return mask.copy()
    return resample_mask(mask, target_shape)


def build_pair_masks(store, pairs):
    """Load every frame and rasterize the strict per-photo observed masks."""
    tiles = {pair.tile_id: store.tile(pair.tile_id) for pair in pairs}
    frames = {}
    observed = {}
    for pair in pairs:
        tile = tiles[pair.tile_id]
        for photo_id in (pair.before_photo_id, pair.after_photo_id):
            frame = specimen_frame(store.blobs, tile, photo_id)
            frames[photo_id] = frame
            observed[photo_id] = pair_observed_mask(tile, pair, frame)
    grid_shape, pixel_scale = canonical_grid(pairs, frames)
    canonical = {photo_id: resample_mask(mask, grid_shape) for photo_id, mask in observed.items()}
    return tiles, frames, canonical, pixel_scale


def common_mask(pairs, canonical):
    mask = None
    for pair in pairs:
        for photo_id in (pair.before_photo_id, pair.after_photo_id):
            mask = canonical[photo_id] if mask is None else (mask & canonical[photo_id])
    return mask


def select_pairs(candidates, canonical, pixel_scale, min_pairs, min_area_mm2):
    """Exact lexicographic selection.

    Stated principle: first keep the largest possible number of fragment
    pairs; among every choice retaining that number, keep the one with the
    largest strictly common visible area. Ties are broken by the ordered
    specimen IDs so the same data always yields one identical study.
    """
    if not 1 <= min_pairs <= len(candidates):
        raise InputError("Minimum pairs must be between one and the number of eligible fragment pairs.")
    if min_area_mm2 < 0:
        raise InputError("Minimum common area cannot be negative.")
    mm_per_pixel = pixel_scale[0] * pixel_scale[1]
    bitsets = [
        np.packbits(canonical[pair.before_photo_id] & canonical[pair.after_photo_id])
        for pair in candidates
    ]
    full = np.packbits(np.ones(canonical[candidates[0].before_photo_id].shape, dtype=np.uint8))
    best = {"count": 0, "area": -1.0, "ids": []}

    def search(start, chosen, shared):
        remaining = len(candidates) - start
        if len(chosen) + remaining < max(min_pairs, best["count"]):
            return
        # Upper bounds on cardinality and on the shared area: adding a pair can
        # only shrink the shared region, so a branch already below the area
        # floor can never satisfy it deeper down.
        shared_pixels = int(np.count_nonzero(np.unpackbits(shared)))
        if shared_pixels * mm_per_pixel + 1e-9 < min_area_mm2:
            return
        if start == len(candidates):
            if len(chosen) < min_pairs:
                return
            area = shared_pixels * mm_per_pixel
            ids = sorted(candidates[index].tile_id for index in chosen)
            if (len(chosen), area) > (best["count"], best["area"]) or (
                len(chosen) == best["count"] and area == best["area"] and ids < best["ids"]
            ):
                best.update(count=len(chosen), area=area, ids=ids)
            return
        # Include the next candidate first so high-cardinality branches prune
        # the rest of the search space quickly.
        search(start + 1, chosen + [start], shared & bitsets[start])
        search(start + 1, chosen, shared)

    search(0, [], full)
    if not best["ids"]:
        raise InputError(
            "No selection reaches both thresholds. Lower the minimum pair count or the minimum common visible area."
        )
    chosen_ids = set(best["ids"])
    selected = [pair for pair in candidates if pair.tile_id in chosen_ids]
    return selected, best["area"]


def coverage_on_study(tile, frame, study_mask, parameters):
    linear, correction = corrected_linear(tile, frame, parameters["apply_neutral"])
    area = mask_to_frame(study_mask, frame)
    unrestricted = region_domain(frame.extent, tile)
    metrics, states = measure_coverage(tile, frame, linear, area, unrestricted, parameters)
    return metrics, states, correction


def pair_row(pair, before_metrics, after_metrics):
    return {
        "tile_id": pair.tile_id,
        "tile_name": pair.tile_name,
        "region_id": pair.region_id,
        "region_name": pair.region_name,
        "before_photo_id": pair.before_photo_id,
        "after_photo_id": pair.after_photo_id,
        "before_bare_percent": before_metrics["percent"]["bare"],
        "after_bare_percent": after_metrics["percent"]["bare"],
        "bare_change_percent": round(after_metrics["percent"]["bare"] - before_metrics["percent"]["bare"], 2),
        "before_glazed_percent": before_metrics["percent"]["glazed"],
        "after_glazed_percent": after_metrics["percent"]["glazed"],
        "before_mixed_percent": before_metrics["percent"]["mixed"],
        "after_mixed_percent": after_metrics["percent"]["mixed"],
        "measured_pixels": before_metrics["measured"],
    }


def study_sheets(store, title, cards):
    digests = []
    for offset in range(0, len(cards), 6):
        group = cards[offset:offset + 6]
        label = title if len(cards) <= 6 else f"{title} · Sheet {offset // 6 + 1}"
        digests.append(stored_preview(comparison_sheet(group, label), store.blobs))
    return digests


def cleaning_candidates(store):
    """Eligible pairs and rejection reasons across the whole collection.

    A specimen with several inspection regions is returned as eligible for
    every region choice, so the UI can offer a region picker rather than
    rejecting it outright.
    """
    eligible = []
    rejected = []
    for tile in store.list_tiles():
        pair, reason, region_options = pair_diagnostics(store, tile)
        if pair is None:
            if reason == "Choose which inspection region defines the study area." and len(region_options) > 1:
                for option in region_options:
                    chosen, second_reason, _ = pair_diagnostics(store, tile, option["id"])
                    if chosen is not None:
                        eligible.append(chosen)
                continue
            rejected.append({"tile_id": tile["id"], "tile_name": tile["name"], "reason": reason})
        else:
            eligible.append(pair)
    if len(eligible) > MAX_CANDIDATE_PAIRS:
        raise InputError(
            f"A cleaning study supports at most {MAX_CANDIDATE_PAIRS} fragment pairs; "
            "narrow the collection or split the work into separate studies."
        )
    return eligible, rejected


def create_cleaning_study(store, payload):
    data = dictionary(payload, "Cleaning study")
    title = string(data.get("title"), "Study title", maximum=100)
    min_pairs = integer(data.get("min_pairs", 2), "Minimum fragment pairs", lower=1, upper=MAX_CANDIDATE_PAIRS)
    min_area = scalar(data.get("min_common_area_mm2", 0), "Minimum common visible area", lower=0, upper=1_000_000)
    apply_neutral = flag(data.get("apply_neutral", False), "Apply neutral-card balance")
    region_choices = data.get("region_ids", {})
    if region_choices is not None and not isinstance(region_choices, dict):
        raise InputError("Region choices must be an object keyed by specimen ID.")
    candidates = []
    rejected = []
    for tile in store.list_tiles():
        chosen_region = (
            identity(region_choices[tile["id"]], "Inspection region")
            if region_choices and tile["id"] in region_choices else None
        )
        pair, reason, _ = pair_diagnostics(store, tile, chosen_region)
        if pair is None:
            rejected.append({"tile_id": tile["id"], "tile_name": tile["name"], "reason": reason})
        else:
            candidates.append(pair)
    if len(candidates) > MAX_CANDIDATE_PAIRS:
        raise InputError(f"A cleaning study supports at most {MAX_CANDIDATE_PAIRS} fragment pairs.")
    if len(candidates) < min_pairs:
        raise InputError(
            f"Only {len(candidates)} fragment pair(s) are eligible, fewer than the minimum of {min_pairs}."
        )
    candidates.sort(key=lambda pair: pair.tile_name.casefold())
    tiles, frames, canonical, pixel_scale = build_pair_masks(store, candidates)
    selected, common_area = select_pairs(candidates, canonical, pixel_scale, min_pairs, min_area)
    study_mask = common_mask(selected, canonical)
    if np.count_nonzero(study_mask) < MIN_COMMON_PIXELS:
        raise InputError("The common visible region has too few supported pixels for a usable study.")
    parameters = {**STUDY_COVERAGE_OPTIONS, "apply_neutral": apply_neutral}
    rows = []
    cards = []
    for pair in selected:
        tile = tiles[pair.tile_id]
        # Each fragment carries its own bare-plaster and intact-pigment
        # controls; the classification settings stay identical across pairs.
        pair_parameters = {
            **parameters,
            "bare_region": unique_reference_region(tile, "bare")["id"],
            "glazed_region": unique_reference_region(tile, "glazed")["id"],
        }
        before_frame = frames[pair.before_photo_id]
        after_frame = frames[pair.after_photo_id]
        before_metrics, before_states, correction = coverage_on_study(
            tile, before_frame, study_mask, pair_parameters
        )
        after_metrics, after_states, _ = coverage_on_study(tile, after_frame, study_mask, pair_parameters)
        rows.append(pair_row(pair, before_metrics, after_metrics))
        cards.append({
            "image": coverage_overlay(before_frame, before_states, mask_to_frame(study_mask, before_frame)),
            "heading": f"{pair.tile_name} · Before",
            "description": f"Bare plaster {before_metrics['percent']['bare']:.2f}%",
        })
        cards.append({
            "image": coverage_overlay(after_frame, after_states, mask_to_frame(study_mask, after_frame)),
            "heading": f"{pair.tile_name} · After",
            "description": f"Bare plaster {after_metrics['percent']['bare']:.2f}%",
        })
    grid_width, grid_height = frames[selected[0].before_photo_id].extent
    mask_digest = store.blobs.put(mask_png(study_mask))
    changes = [row["bare_change_percent"] for row in rows]
    study = {
        "study_type": "cleaning",
        "title": title,
        "selection_rule": (
            "Lexicographic rule: retain the maximum possible number of fragment pairs first, "
            "then among those choices maximize the common visible area; ties resolve by specimen ID."
        ),
        "thresholds": {"min_pairs": min_pairs, "min_common_area_mm2": min_area},
        "parameters": parameters,
        "plane": {
            "width_px": grid_width,
            "height_px": grid_height,
            "width_mm": selected[0].width_mm,
            "height_mm": selected[0].height_mm,
            "pixel_scale_mm": [round(pixel_scale[0], 8), round(pixel_scale[1], 8)],
        },
        "common_mask_digest": mask_digest,
        "common_area_mm2": round(common_area, 3),
        "common_pixels": int(np.count_nonzero(study_mask)),
        "pairs": [
            {
                "tile_id": pair.tile_id,
                "tile_name": pair.tile_name,
                "region_id": pair.region_id,
                "region_name": pair.region_name,
                "before_photo_id": pair.before_photo_id,
                "after_photo_id": pair.after_photo_id,
                "photo_digests": {
                    "before": frames[pair.before_photo_id].photo["source_digest"],
                    "after": frames[pair.after_photo_id].photo["source_digest"],
                },
            }
            for pair in selected
        ],
        "rows": rows,
        "groups": study_stats([
            {"group": "All fragments", "value": row["bare_change_percent"]}
            for row in rows
        ]),
        "bare_change": {
            "mean_percent": round(float(np.mean(changes)), 3),
            "minimum_percent": round(float(min(changes)), 3),
            "maximum_percent": round(float(max(changes)), 3),
        },
        "excluded_specimens": rejected,
        "candidate_count": len(candidates),
        "frozen_note": (
            "This study is frozen: the adopted pairs, common region and photo set never change when "
            "new photos are uploaded or independent measurements are saved."
        ),
        "pictures": study_sheets(store, title, cards),
    }
    return store.save_study(study)


def study_for_tile(store, study_id, tile_id):
    study = store.study(study_id)
    if study.get("study_type") != "cleaning":
        raise InputError("Choose a cleaning study for this re-check.")
    entry = next((item for item in study["pairs"] if item["tile_id"] == tile_id), None)
    if entry is None:
        raise InputError("This specimen was not retained by the chosen cleaning study.")
    return study, entry


def check_region(payload):
    data = dictionary(payload, "Re-check region")
    if data.get("region_id"):
        return identity(data["region_id"], "Re-check region"), None
    vertices = data.get("vertices")
    if not vertices:
        raise InputError("Choose a saved region or draw the small re-check box.")
    if not isinstance(vertices, list) or not 3 <= len(vertices) <= 40:
        raise InputError("The re-check box needs between three and forty vertices.")
    points = []
    for vertex in vertices:
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise InputError("Each re-check vertex needs two coordinates.")
        points.append([
            scalar(vertex[0], "Vertex X", lower=0, upper=1),
            scalar(vertex[1], "Vertex Y", lower=0, upper=1),
        ])
    return None, points


def create_cleaning_check(store, tile_id, payload):
    study_id = identity(payload.get("study_id"), "Cleaning study")
    study, entry = study_for_tile(store, study_id, tile_id)
    tile = store.tile(tile_id)
    region_id, vertices = check_region(payload)
    parameters = {
        **STUDY_COVERAGE_OPTIONS,
        "apply_neutral": flag(payload.get("apply_neutral", False), "Apply neutral-card balance"),
        "bare_region": unique_reference_region(tile, "bare")["id"],
        "glazed_region": unique_reference_region(tile, "glazed")["id"],
    }
    before_frame = specimen_frame(store.blobs, tile, entry["before_photo_id"])
    after_frame = specimen_frame(store.blobs, tile, entry["after_photo_id"])
    study_mask = load_mask(store.blobs, study["common_mask_digest"])
    region = None
    if region_id:
        region = next((item for item in tile["regions"] if item["id"] == region_id), None)
        if region is None or region["purpose"] == "ignore":
            raise InputError("Choose a valid saved region for the re-check.")
        box = polygon_mask(before_frame.extent, region["vertices"])
        region_name = region["name"]
    else:
        box = polygon_mask(before_frame.extent, vertices)
        region_name = "Drawn re-check box"
    # The small box is measurable only where it intersects the adopted region
    # and only at positions the study's photos actually observed.
    allowed = box & mask_to_frame(study_mask, before_frame)
    allowed = allowed & mask_to_frame(study_mask, after_frame)
    if np.count_nonzero(allowed) < MIN_COMMON_PIXELS:
        raise InputError(
            "The re-check box barely intersects the adopted common visible region. "
            "Only positions observed by the original study can be measured."
        )
    before_metrics, before_states, correction = coverage_on_study(tile, before_frame, allowed, parameters)
    after_metrics, after_states, _ = coverage_on_study(tile, after_frame, allowed, parameters)
    before_overlay = stored_preview(coverage_overlay(before_frame, before_states, allowed), store.blobs)
    after_overlay = stored_preview(
        coverage_overlay(after_frame, after_states, mask_to_frame(allowed, after_frame)), store.blobs
    )
    requested_pixels = int(np.count_nonzero(box))
    result = {
        "tile_id": tile_id,
        "source_version": tile["version"],
        "kind": "cleaning-check",
        "label": "Cleaning study re-check",
        "parameters": {
            "kind": "cleaning-check",
            "options": parameters,
            "study_id": study_id,
            "region_id": region_id,
            "vertices": vertices,
        },
        "metrics": {
            "study_title": study["title"],
            "study_id": study_id,
            "region_name": region_name,
            "requested_box_pixels": requested_pixels,
            "eligible_pixels": int(np.count_nonzero(allowed)),
            "eligible_percent_of_box": round(100 * np.count_nonzero(allowed) / max(requested_pixels, 1), 2),
            "before": {
                "percent": before_metrics["percent"],
                "area_mm2": before_metrics["area_mm2"],
                "measured": before_metrics["measured"],
            },
            "after": {
                "percent": after_metrics["percent"],
                "area_mm2": after_metrics["area_mm2"],
                "measured": after_metrics["measured"],
            },
            "bare_change_percent": round(after_metrics["percent"]["bare"] - before_metrics["percent"]["bare"], 2),
        },
        "legend": {
            "mode": "categorical",
            "labels": [
                {"name": "Bare plaster", "color": "#ebb25e"},
                {"name": "Mixed", "color": "#dad892"},
                {"name": "Intact pigment", "color": "#5ba88f"},
                {"name": "Unmeasured", "color": "#353a3a"},
            ],
        },
        "basis": {
            "photo_role": "before",
            "synthetic": bool(before_frame.photo.get("synthetic")),
            "neutral_target": correction["target"] if correction else None,
            "source_space": before_frame.photo["working_color_space"],
            "tile_id": tile_id,
        },
        "support": {
            "registered_image_extent": list(before_frame.extent),
            "requested_pixels": requested_pixels,
            "available_pixels": int(np.count_nonzero(allowed)),
            "clipped_pixels": 0,
            "available_percent": round(100 * np.count_nonzero(allowed) / max(requested_pixels, 1), 2),
            "calibrated_area_mm2": (
                round(np.count_nonzero(allowed) * before_frame.scale[0] * before_frame.scale[1], 3)
                if before_frame.scale else None
            ),
        },
        "provenance": {
            "analyzer_version": 1,
            "specimen": {
                "id": tile["id"],
                "version": tile["version"],
                "name": tile["name"],
                "body": tile["body"],
                "glaze": tile["glaze"],
                "firing": tile["firing"],
                "batch": tile["batch"],
                "physical_mm": [tile["width_mm"], tile["height_mm"]],
            },
            "photo": {
                "id": before_frame.photo["id"],
                "display_name": before_frame.photo["name"],
                "role": before_frame.photo["role"],
                "original_digest": before_frame.photo["source_digest"],
                "working_digest": before_frame.photo["working_digest"],
                "raw_size": before_frame.photo["original_size"],
                "upright_size": [before_frame.photo["width"], before_frame.photo["height"]],
                "working_color_space": before_frame.photo["working_color_space"],
                "plane_to_capture": list(before_frame.transform),
                "synthetic": before_frame.photo["synthetic"],
            },
            "paired_photo": {
                "id": after_frame.photo["id"],
                "display_name": after_frame.photo["name"],
                "role": after_frame.photo["role"],
                "original_digest": after_frame.photo["source_digest"],
                "working_digest": after_frame.photo["working_digest"],
                "raw_size": after_frame.photo["original_size"],
                "upright_size": [after_frame.photo["width"], after_frame.photo["height"]],
                "working_color_space": after_frame.photo["working_color_space"],
                "plane_to_capture": list(after_frame.transform),
                "synthetic": after_frame.photo["synthetic"],
            },
            "region": (
                {"id": region_id, "name": region_name}
                if region_id else {"id": None, "name": region_name}
            ),
            "exclusions": [item for item in tile["regions"] if item["purpose"] == "ignore"],
            "neutral_card": correction,
            "reference": None,
            "study": {
                "id": study_id,
                "common_mask_digest": study["common_mask_digest"],
                "common_area_mm2": study["common_area_mm2"],
                "frozen": True,
            },
        },
        "pictures": {
            "source": stored_preview(base_rgba(before_frame), store.blobs),
            "overlay": before_overlay,
            "paired": after_overlay,
        },
    }
    cache_key = key_for({
        "kind": "cleaning-check",
        "study_id": study_id,
        "tile_id": tile_id,
        "region_id": region_id,
        "region_vertices": region["vertices"] if region_id else None,
        "vertices": vertices,
        "parameters": parameters,
        "mask_digest": study["common_mask_digest"],
        "before_photo_id": entry["before_photo_id"],
        "after_photo_id": entry["after_photo_id"],
    })
    return store.save_result(cache_key, result)
