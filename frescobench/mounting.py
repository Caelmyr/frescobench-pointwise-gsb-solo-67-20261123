"""Choose one pair of mounting foot pads and classify later footing traces.

Candidates, bearing patches, no-pressure zones and visible painted areas are
marked as sampling regions on the measured plane.  Adoption freezes the chosen
outlines and the regions around them, so later region work cannot move the pair
an installer is given.  A footing trace is read against that frozen pair only.
"""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .analysis import capture_provenance, specimen_provenance
from .art import SHADOW, base_rgba, font, stored_preview
from .geometry import (
    polygon_mask,
    signed_area,
    specimen_frame,
    validate_polygon,
)
from .metrology import path_cumulative, path_segments
from .model import InputError, collection, dictionary, numeric_pair, scalar, string, timestamp
from .store import make_id


SIDE_PURPOSES = {"A": "pad_a", "B": "pad_b"}
SIDE_LABELS = {"A": "Pad A", "B": "Pad B"}
PAD_COLORS = {"A": (246, 139, 96), "B": (112, 168, 219)}
BEARING_COLORS = {"A": (102, 205, 143), "B": (86, 191, 207)}
VISIBLE_COLOR = (232, 226, 189)
NO_PRESSURE_COLOR = (238, 121, 116)
OPEN_GAP_COLOR = (235, 126, 102)
INK = (255, 245, 225)
RULE_KEY = "paint_first_then_bearing"
RULE_TEXT = (
    "Painting first: among pairs that meet both bearing minimums, the visible-paint "
    "minimum, the no-overlap rule and the no-pressure zone, adopt the pair covering the "
    "least visible painted area. Equal paint coverage goes to the larger combined "
    "acknowledged bearing area, then to the earlier-marked candidate on each side. "
    "Minimums are never traded for extra bearing."
)
COMPONENT_STRUCTURE = np.ones((3, 3), dtype=int)
MIN_BEARING_COMPONENT_PIXELS = 4
MIN_TRACE_POINTS = 2
MAX_TRACE_POINTS = 45
MAX_TRACE_SAMPLES = 2400


@dataclass
class Candidate:
    side: str
    region_id: str
    name: str
    vertices: list
    order: int
    mask: np.ndarray
    acknowledged: np.ndarray | None
    bearing_pixels: int
    bearing_reason: str | None
    visible_pixels: int
    no_pressure_pixels: int


@dataclass
class FrozenPlan:
    frame: object
    plan: dict
    pads: dict
    pad_masks: dict
    acknowledged: dict


def frozen_regions(tile, selected):
    """Snapshot the adopted pads and bearing areas; rejected candidates stay editable."""
    chosen = {selected["A"].region_id, selected["B"].region_id}
    return [
        region_snapshot(region) for region in tile["regions"]
        if region["purpose"] == "bearing" or region["id"] in chosen
    ]


def region_snapshot(region):
    return {
        "id": region["id"],
        "name": region["name"],
        "purpose": region["purpose"],
        "vertices": [list(vertex) for vertex in region["vertices"]],
    }


def union_mask(extent, regions):
    width, height = extent
    mask = np.zeros((height, width), dtype=bool)
    for region in regions:
        mask |= polygon_mask(extent, region["vertices"])
    return mask


def evident_pixels(frame):
    return frame.valid & ~frame.clipped


def area_units(pixels, scale):
    """Express a pixel count in mm² when the plane has a physical scale."""
    if not scale:
        return round(float(pixels), 4)
    return round(float(pixels) * float(scale[0]) * float(scale[1]), 4)


def plan_unit(frame):
    return "mm²" if frame.scale else "px²"


def largest_evident_component(pad_mask, bearing_any, evident):
    support = pad_mask & bearing_any & evident
    labels, count = ndimage.label(support, structure=COMPONENT_STRUCTURE)
    if not count:
        return None, 0, "no evident bearing pixels"
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    chosen = int(np.argmax(sizes))
    if sizes[chosen] < MIN_BEARING_COMPONENT_PIXELS:
        return None, 0, "only isolated bearing pixels"
    return labels == chosen, int(sizes[chosen]), None


def candidate_for(frame, region, side, order, bearing_any, visible_any, no_pressure_any):
    mask = polygon_mask(frame.extent, region["vertices"])
    evident = evident_pixels(frame)
    acknowledged, bearing_pixels, reason = largest_evident_component(mask, bearing_any, evident)
    return Candidate(
        side=side,
        region_id=region["id"],
        name=region["name"],
        vertices=[list(vertex) for vertex in region["vertices"]],
        order=order,
        mask=mask,
        acknowledged=acknowledged,
        bearing_pixels=bearing_pixels,
        bearing_reason=reason,
        visible_pixels=int(np.count_nonzero(mask & visible_any & evident)),
        no_pressure_pixels=int(np.count_nonzero(mask & no_pressure_any)),
    )


def candidates_for(tile, frame, side, bearing_any, visible_any, no_pressure_any):
    purpose = SIDE_PURPOSES[side]
    chosen = []
    for order, region in enumerate(region for region in tile["regions"] if region["purpose"] == purpose):
        validate_polygon(region["vertices"])
        chosen.append(candidate_for(frame, region, side, order, bearing_any, visible_any, no_pressure_any))
    return chosen


def pair_sort_key(first, second, covered_visible):
    return (
        covered_visible,
        -(first.bearing_pixels + second.bearing_pixels),
        first.order,
        second.order,
    )


def pair_obstacle(first, second, limits):
    if first.no_pressure_pixels or second.no_pressure_pixels:
        return "no-pressure zone"
    if np.any(first.mask & second.mask):
        return "pads overlap"
    if first.bearing_pixels < limits["bearing_a"]:
        return "Pad A bearing minimum"
    if second.bearing_pixels < limits["bearing_b"]:
        return "Pad B bearing minimum"
    return None


def adoption_failure(counts, blockers, visible_missing):
    details = []
    if counts["A"] == 0:
        details.append("no side A candidate is marked")
    if counts["B"] == 0:
        details.append("no side B candidate is marked")
    if details:
        raise InputError("Mark candidate pad outlines first: " + "; ".join(details) + ".")
    reasons = ", ".join(f"{count} blocked by {name}" for name, count in blockers.items() if count)
    if visible_missing:
        reasons = ", ".join(part for part in (reasons, f"{visible_missing} blocked by the visible-paint minimum") if part)
    raise InputError(
        "No pad pair meets the entered minimums, the no-overlap rule and the no-pressure zone. "
        + (f"Checked pairs: {reasons}." if reasons else "Check the marked bearing and visible areas.")
    )

def choose_pair(tile, frame, minimums):
    extent = frame.extent
    bearing_any = union_mask(extent, [region for region in tile["regions"] if region["purpose"] == "bearing"])
    visible_any = union_mask(extent, [region for region in tile["regions"] if region["purpose"] == "visible"])
    no_pressure_any = union_mask(extent, [region for region in tile["regions"] if region["purpose"] == "nopressure"])
    if not bearing_any.any():
        raise InputError("Mark at least one supportable bare plaster area before adopting pads.")
    if minimums["visible"] > 0 and not visible_any.any():
        raise InputError("Mark the painted area that must stay visible before setting a visible-paint minimum.")
    evident = evident_pixels(frame)
    visible_total = int(np.count_nonzero(visible_any & evident))
    firsts = candidates_for(tile, frame, "A", bearing_any, visible_any, no_pressure_any)
    seconds = candidates_for(tile, frame, "B", bearing_any, visible_any, no_pressure_any)
    if not firsts or not seconds:
        adoption_failure({"A": len(firsts), "B": len(seconds)}, {}, 0)
    scale = frame.scale
    limits = {
        "bearing_a": minimums["bearing_a"] / (float(scale[0]) * float(scale[1])) if scale else minimums["bearing_a"],
        "bearing_b": minimums["bearing_b"] / (float(scale[0]) * float(scale[1])) if scale else minimums["bearing_b"],
    }
    visible_limit = minimums["visible"] / (float(scale[0]) * float(scale[1])) if scale else minimums["visible"]
    blockers = {}
    visible_missing = 0
    feasible = []
    for first in firsts:
        for second in seconds:
            reason = pair_obstacle(first, second, limits)
            if reason is not None:
                blockers[reason] = blockers.get(reason, 0) + 1
                continue
            remaining = visible_total - first.visible_pixels - second.visible_pixels
            if remaining < visible_limit:
                visible_missing += 1
                continue
            feasible.append((first, second, first.visible_pixels + second.visible_pixels, remaining))
    if not feasible:
        adoption_failure({"A": len(firsts), "B": len(seconds)}, blockers, visible_missing)
    feasible.sort(key=lambda item: pair_sort_key(item[0], item[1], item[2]))
    first, second, covered, remaining = feasible[0]
    return {
        "A": first,
        "B": second,
        "bearing_any": bearing_any,
        "visible_any": visible_any,
        "no_pressure_any": no_pressure_any,
        "visible_total": visible_total,
        "covered_visible": covered,
        "remaining_visible": remaining,
        "feasible_pairs": len(feasible),
        "candidate_counts": {"A": len(firsts), "B": len(seconds)},
    }


def pad_record(candidate):
    return {
        "region_id": candidate.region_id,
        "name": candidate.name,
        "vertices": candidate.vertices,
        "order": candidate.order,
        "outline_area": round(abs(signed_area(candidate.vertices)), 6),
    }


def acknowledged_record(candidate, scale):
    return {
        "pixels": candidate.bearing_pixels,
        "area": area_units(candidate.bearing_pixels, scale),
        "recognized_pieces": 0 if candidate.acknowledged is None else 1,
        "note": "One connected evident bearing piece is acknowledged; separated areas are not added.",
    }


def tint(image_array, mask, color, strength):
    if mask is None or not mask.any():
        return
    image_array[mask, :3] = image_array[mask, :3] * (1 - strength) + np.asarray(color, dtype=np.float32) * strength


def boundary(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=COMPONENT_STRUCTURE)


def draw_plan(frame, selected, unit):
    image = base_rgba(frame)
    raw = np.asarray(image, dtype=np.float32).copy()
    tint(raw, selected["no_pressure_any"], NO_PRESSURE_COLOR, 0.34)
    tint(raw, selected["visible_any"], VISIBLE_COLOR, 0.16)
    tint(raw, selected["bearing_any"], (150, 190, 140), 0.20)
    for side in ("A", "B"):
        tint(raw, selected[side].acknowledged, BEARING_COLORS[side], 0.55)
    image = Image.fromarray(raw.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke = max(2, round(min(width, height) / 260))
    for side in ("A", "B"):
        candidate = selected[side]
        points = [(round(x * width), round(y * height)) for x, y in candidate.vertices]
        draw.line([*points, points[0]], fill=SHADOW, width=stroke + 3, joint="curve")
        draw.line([*points, points[0]], fill=PAD_COLORS[side], width=stroke + 1, joint="curve")
        if candidate.acknowledged is not None:
            edges = boundary(candidate.acknowledged)
            raw[edges, :3] = np.asarray(BEARING_COLORS[side], dtype=np.float32)
            image = Image.fromarray(raw.astype(np.uint8))
            draw = ImageDraw.Draw(image)
        x, y = points[0]
        label = f"{SIDE_LABELS[side]} · {area_units(candidate.bearing_pixels, frame.scale):.2f} {unit}"
        draw.text((x + 5, y + 4), label, fill=PAD_COLORS[side], stroke_width=2, stroke_fill=SHADOW)
    typeface = font(max(11, round(min(width, height) / 48)))
    draw.text((8, height - 62), "Adopted mounting pair · one connected bearing piece per pad", font=typeface,
              fill=INK, stroke_width=3, stroke_fill=SHADOW)
    draw.text((8, height - 34), "Green/teal: acknowledged bearing · red tint: no-pressure zone",
              font=typeface, fill=INK, stroke_width=3, stroke_fill=SHADOW)
    return image


def adopt_mounting(store, specimen_id, payload):
    data = dictionary(payload, "Mounting pair")
    tile = store.tile(specimen_id)
    expected = data.get("version")
    if type(expected) is not int or expected < 1:
        raise InputError("Version must be a whole number of at least 1.")
    store.require_version(tile, expected)
    frame = specimen_frame(store.blobs, tile)
    minimums = {
        "bearing_a": scalar(data.get("min_bearing_a"), "Pad A minimum bearing area", lower=0),
        "bearing_b": scalar(data.get("min_bearing_b"), "Pad B minimum bearing area", lower=0),
        "visible": scalar(data.get("min_visible"), "Minimum remaining visible paint area", lower=0),
    }
    selected = choose_pair(tile, frame, minimums)
    unit = plan_unit(frame)
    plan = {
        "id": make_id(),
        "tile_id": tile["id"],
        "photo_id": tile["plane"]["photo_id"],
        "plane": {
            "photo_id": tile["plane"]["photo_id"],
            "corners": [list(vertex) for vertex in tile["plane"]["corners"]],
            "size": list(tile["plane"]["size"]),
        },
        "created_at": timestamp(),
        "unit": unit,
        "scale": list(frame.scale) if frame.scale else None,
        "minimums": minimums,
        "rule": {"key": RULE_KEY, "text": RULE_TEXT},
        "candidate_counts": selected["candidate_counts"],
        "feasible_pairs": selected["feasible_pairs"],
        "regions": frozen_regions(tile, selected),
        "pads": {"A": pad_record(selected["A"]), "B": pad_record(selected["B"])},
        "acknowledged": {
            side: acknowledged_record(selected[side], frame.scale) for side in ("A", "B")
        },
        "visible": {
            "evident_pixels": selected["visible_total"],
            "evident_area": area_units(selected["visible_total"], frame.scale),
            "covered_pixels": selected["covered_visible"],
            "covered_area": area_units(selected["covered_visible"], frame.scale),
            "remaining_pixels": selected["remaining_visible"],
            "remaining_area": area_units(selected["remaining_visible"], frame.scale),
        },
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frame),
        "image": stored_preview(draw_plan(frame, selected, unit), store.blobs),
    }
    return store.save_mounting(specimen_id, expected, plan)


def release_mounting(store, specimen_id, payload):
    data = dictionary(payload, "Mounting release")
    expected = data.get("version")
    if type(expected) is not int or expected < 1:
        raise InputError("Version must be a whole number of at least 1.")
    tile = store.tile(specimen_id)
    store.require_version(tile, expected)

    def change(current):
        current["mounting"] = None

    return store.edit_tile(specimen_id, expected, change, "mounting", "Released the adopted mounting pair")


def plan_status(tile):
    plan = tile.get("mounting")
    if not plan:
        return None
    current = {region["id"]: region for region in tile["regions"]}
    frozen = {region["id"]: region for region in plan["regions"]}
    missing = sorted(
        snapshot["name"] for region_id, snapshot in frozen.items()
        if region_id not in current or current[region_id]["purpose"] != snapshot["purpose"]
    )
    changed = sorted(
        snapshot["name"] for region_id, snapshot in frozen.items()
        if region_id in current
        and current[region_id]["purpose"] == snapshot["purpose"]
        and [list(vertex) for vertex in current[region_id]["vertices"]] != snapshot["vertices"]
    )
    plane = tile.get("plane")
    plane_changed = (
        plane is None
        or plane["photo_id"] != plan["photo_id"]
        or [list(vertex) for vertex in plane["corners"]] != plan["plane"]["corners"]
        or list(plane["size"]) != plan["plane"]["size"]
    )
    return {
        "stale": bool(missing or changed or plane_changed),
        "missing": missing,
        "changed": changed,
        "plane_changed": plane_changed,
    }


def retained_regions(plan, purpose):
    return [
        {"id": region["id"], "name": region["name"], "purpose": region["purpose"], "vertices": region["vertices"]}
        for region in plan["regions"] if region["purpose"] == purpose
    ]


def frozen_plan(store, tile):
    plan = tile.get("mounting")
    if plan is None:
        raise InputError("Adopt a mounting pair before tracing a footing line.")
    status = plan_status(tile)
    if status["plane_changed"]:
        raise InputError("The specimen plane moved since the mounting pair was adopted. Re-adopt the pair before tracing.")
    if status["stale"]:
        names = ", ".join(status["missing"] + status["changed"])
        raise InputError(
            f"The adopted mounting pair no longer matches its frozen regions ({names}). Re-adopt the pair before tracing."
        )
    plane = tile["plane"]
    frame = specimen_frame(store.blobs, tile, plan["photo_id"])
    bearing_any = union_mask(frame.extent, retained_regions(plan, "bearing"))
    acknowledged = {}
    pads = {}
    pad_masks = {}
    for side in ("A", "B"):
        pads[side] = plan["pads"][side]
        mask = polygon_mask(frame.extent, pads[side]["vertices"])
        pad_masks[side] = mask
        chosen, pixels, reason = largest_evident_component(mask, bearing_any, evident_pixels(frame))
        if chosen is None or pixels != plan["acknowledged"][side]["pixels"]:
            raise InputError(
                f"The frozen {SIDE_LABELS[side]} bearing piece no longer reproduces. Re-adopt the mounting pair."
            )
        acknowledged[side] = chosen
    return FrozenPlan(frame, plan, pads, pad_masks, acknowledged)


def trace_points(data):
    raw = collection(data.get("points"), "Footing line points", minimum=MIN_TRACE_POINTS, maximum=MAX_TRACE_POINTS)
    return [numeric_pair(item, "Footing line point", lower=0, upper=1) for item in raw]


def pixel_positions(points, extent):
    return np.asarray(points, dtype=np.float64) * np.asarray(extent, dtype=np.float64) - 0.5


def sample_line(frozen, points):
    frame = frozen.frame
    height, width = frame.valid.shape
    positions = pixel_positions(points, frame.extent)
    scale = np.asarray(frozen.frame.scale or (1.0, 1.0), dtype=np.float64)
    codes = []
    distances = []
    station_points = []
    travelled = 0.0
    for start, stop in zip(positions, positions[1:]):
        delta = stop - start
        length = float(np.linalg.norm(delta * scale))
        count = min(MAX_TRACE_SAMPLES, max(2, int(math.ceil(float(np.linalg.norm(delta)))) + 1))
        for index in range(count):
            fraction = index / (count - 1)
            x, y = start + fraction * delta
            col = int(np.clip(round(x), 0, width - 1))
            row = int(np.clip(round(y), 0, height - 1))
            codes.append(line_code(frozen, row, col))
            distances.append(travelled + fraction * length)
            station_points.append((x, y))
        travelled += length
    end = positions[-1]
    col = int(np.clip(round(end[0]), 0, width - 1))
    row = int(np.clip(round(end[1]), 0, height - 1))
    codes.append(line_code(frozen, row, col))
    distances.append(travelled)
    station_points.append((float(end[0]), float(end[1])))
    return codes, distances, station_points


def line_code(frozen, row, col):
    if frozen.acknowledged["A"] is not None and frozen.acknowledged["A"][row, col]:
        return "A"
    if frozen.acknowledged["B"] is not None and frozen.acknowledged["B"][row, col]:
        return "B"
    if frozen.pad_masks["A"][row, col]:
        return "under_a"
    if frozen.pad_masks["B"][row, col]:
        return "under_b"
    return "open"


RUN_LABELS = {
    "A": "continuous · Pad A bearing",
    "B": "continuous · Pad B bearing",
    "under_a": "interruption · under Pad A outside recognized bearing",
    "under_b": "interruption · under Pad B outside recognized bearing",
    "open": "interruption · between pads or outside pads",
}


def classify_runs(codes, distances, station_points, extent, unit):
    runs = []
    start = 0
    for index in range(1, len(codes) + 1):
        if index < len(codes) and codes[index] == codes[start]:
            continue
        code = codes[start]
        first_point = station_points[start]
        last_point = station_points[index - 1]
        runs.append({
            "kind": "continuous" if code in ("A", "B") else "interruption",
            "pad": code if code in ("A", "B") else None,
            "code": code,
            "label": RUN_LABELS[code],
            "from_distance": round(float(distances[start]), 4),
            "to_distance": round(float(distances[index - 1]), 4),
            "length": round(float(distances[index - 1] - distances[start]), 4),
            "first_station": start,
            "last_station": index - 1,
            "from": normalized_point(first_point, extent),
            "to": normalized_point(last_point, extent),
            "unit": unit,
        })
        start = index
    continuous = [run for run in runs if run["kind"] == "continuous"]
    interruptions = [run for run in runs if run["kind"] == "interruption"]
    return runs, {
        "continuous_runs": len(continuous),
        "interruption_runs": len(interruptions),
        "continuous_length": round(sum(run["length"] for run in continuous), 4),
        "interruption_length": round(sum(run["length"] for run in interruptions), 4),
        "pad_a_length": round(sum(run["length"] for run in continuous if run["pad"] == "A"), 4),
        "pad_b_length": round(sum(run["length"] for run in continuous if run["pad"] == "B"), 4),
        "observed_percent": round(
            100 * sum(run["length"] for run in continuous)
            / max(sum(run["length"] for run in runs), 1e-9), 3,
        ),
    }


def normalized_point(point, extent):
    width, height = extent
    return [round(float((point[0] + 0.5) / width), 6), round(float((point[1] + 0.5) / height), 6)]


def draw_footing(frozen, runs):
    image = base_rgba(frozen.frame)
    raw = np.asarray(image, dtype=np.float32).copy()
    for side in ("A", "B"):
        tint(raw, frozen.acknowledged[side], BEARING_COLORS[side], 0.55)
    image = Image.fromarray(raw.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    stroke = max(2, round(min(width, height) / 260))
    for side in ("A", "B"):
        points = [(round(x * width), round(y * height)) for x, y in frozen.pads[side]["vertices"]]
        draw.line([*points, points[0]], fill=SHADOW, width=stroke + 3, joint="curve")
        draw.line([*points, points[0]], fill=PAD_COLORS[side], width=stroke + 1, joint="curve")
    for run in runs:
        start = (run["from"][0] * width, run["from"][1] * height)
        stop = (run["to"][0] * width, run["to"][1] * height)
        if run["kind"] == "continuous":
            color = BEARING_COLORS[run["pad"]]
            draw.line((start, stop), fill=SHADOW, width=stroke + 4)
            draw.line((start, stop), fill=color, width=stroke + 2)
        else:
            draw.line((start, stop), fill=SHADOW, width=stroke + 2)
            draw.line((start, stop), fill=OPEN_GAP_COLOR, width=stroke)
    typeface = font(max(11, round(min(width, height) / 48)))
    continuous = sum(run["length"] for run in runs if run["kind"] == "continuous")
    interrupted = sum(run["length"] for run in runs if run["kind"] == "interruption")
    draw.text((8, height - 62), f"Recognized bearing along line: {continuous:.2f}", font=typeface,
              fill=INK, stroke_width=3, stroke_fill=SHADOW)
    draw.text((8, height - 34), f"Interrupted: {interrupted:.2f} · gaps are not bridged", font=typeface,
              fill=INK, stroke_width=3, stroke_fill=SHADOW)
    return image


def trace_footing(store, specimen_id, payload):
    data = dictionary(payload, "Footing trace")
    title = string(data.get("title"), "Footing trace title", maximum=90)
    tile = store.tile(specimen_id)
    frozen = frozen_plan(store, tile)
    points = trace_points(data)
    segments, unit = path_segments(frozen.frame, points)
    cumulative = path_cumulative(segments)
    codes, distances, station_points = sample_line(frozen, points)
    runs, totals = classify_runs(codes, distances, station_points, frozen.frame.extent, unit)
    overlay = draw_footing(frozen, runs)
    shape = {
        "points": points,
        "segments": segments,
        "cumulative": cumulative,
        "trace_length": cumulative[-1],
        "length_unit": unit,
        "unit": frozen.plan["unit"],
        "runs": runs,
        "totals": totals,
        "minimums": frozen.plan["minimums"],
        "rule": frozen.plan["rule"],
        "acknowledged": frozen.plan["acknowledged"],
        "plan_id": frozen.plan["id"],
        "plan_adopted_at": frozen.plan["created_at"],
        "note": (
            "Continuous runs follow one acknowledged bearing piece only. The line is interrupted across "
            "the gap between pads and wherever a pad carries no recognized bearing, so those lengths are not added."
        ),
    }
    return store.save_metrology({
        "title": title,
        "mode": "footing",
        "tile_id": tile["id"],
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frozen.frame),
        "region": None,
        "exclusions": [region for region in tile["regions"] if region["purpose"] == "ignore"],
        "mounting": {
            "id": frozen.plan["id"],
            "created_at": frozen.plan["created_at"],
            "minimums": frozen.plan["minimums"],
            "rule": frozen.plan["rule"],
        },
        "shape": shape,
        "image": stored_preview(overlay, store.blobs),
    })
