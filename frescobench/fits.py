"""Trial-fit service: payload parsing, evaluation orchestration and previews."""
import numpy as np
from PIL import ImageDraw

from .art import INK, SHADOW, font, stored_preview
from .fitting import (
    Fragment,
    choose_base_map,
    evaluate_candidate,
)
from .model import (
    InputError,
    collection,
    dictionary,
    identity,
    integer,
    numeric_pair,
    scalar,
    string,
    timestamp,
)
from .seams import base_map_image, seam_overlay
from .store import key_for, make_id


SIDES = ("A", "B")
MAX_OUTLINE_POINTS = 200
MAX_CANDIDATES = 4
MAX_COLOR_LINES = 12
CANVAS_LONGEST_PX = 1600


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def _point_list(payload, label, *, minimum, maximum):
    raw = collection(payload, label, minimum=minimum, maximum=maximum)
    return [numeric_pair(item, f"{label} point", lower=0, upper=1) for item in raw]


def _chain_map(payload, label):
    data = dictionary(payload, label)
    chains = {}
    for kind in ("solid", "broken"):
        polylines = collection(data.get(kind), f"{label} {kind} edges", minimum=1, maximum=12)
        chains[kind] = [
            _point_list(polyline, f"{label} {kind} edge", minimum=2, maximum=60)
            for polyline in polylines
        ]
    return chains


def parse_marks(payload, label):
    data = dictionary(payload, label)
    return {
        "outline": _point_list(data.get("outline"), f"{label} outline", minimum=3, maximum=MAX_OUTLINE_POINTS),
        "edges": _chain_map(data.get("edges"), f"{label} edges"),
        "color_lines": [
            _point_list(line, f"{label} colour line", minimum=2, maximum=2)
            for line in collection(data.get("color_lines", []), f"{label} colour lines", minimum=1, maximum=MAX_COLOR_LINES)
        ],
    }


def parse_candidate(payload, index):
    data = dictionary(payload, "Candidate placement")
    candidate_id = string(data.get("id", f"candidate-{index}"), "Candidate ID", maximum=40)
    return {
        "id": candidate_id,
        "label": string(data.get("label", candidate_id), "Candidate label", maximum=80),
        "anchors_a": _point_list(data.get("anchors_a"), "Candidate anchors on fragment A", minimum=2, maximum=2),
        "anchors_b": _point_list(data.get("anchors_b"), "Candidate anchors on fragment B", minimum=2, maximum=2),
    }


def parse_thresholds(payload):
    data = dictionary(payload, "Fit thresholds")
    minimum_match = scalar(data.get("minimum_matched_mm"), "Minimum matchable broken-edge length", lower=0.01, upper=100000)
    maximum_offset = scalar(data.get("maximum_line_offset_mm"), "Maximum colour-line offset at the break", lower=0.001, upper=100000)
    seam_gap = scalar(data.get("seam_gap_mm", 0.0), "Empty seam width at the fracture", lower=0.0, upper=1000.0)
    return {
        "minimum_matched_mm": minimum_match,
        "maximum_line_offset_mm": maximum_offset,
        "seam_gap_mm": seam_gap,
    }


def parse_fit_payload(payload):
    data = dictionary(payload, "Trial fit")
    members = dictionary(data.get("fragments"), "Trial fit fragments")
    fragment_refs = {}
    for side in SIDES:
        entry = dictionary(members.get(side), f"Fragment {side}")
        fragment_refs[side] = {
            "tile_id": identity(entry.get("tile_id"), f"Fragment {side} specimen ID"),
            "photo_id": identity(entry.get("photo_id"), f"Fragment {side} photo ID"),
            "marks": parse_marks(entry.get("marks"), f"Fragment {side} marks"),
        }
    candidates = [
        parse_candidate(item, index + 1)
        for index, item in enumerate(collection(data.get("candidates"), "Candidate placements", minimum=2, maximum=MAX_CANDIDATES))
    ]
    candidate_ids = [item["id"] for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise InputError("Candidate placements must have unique IDs.")
    return {
        "title": string(data.get("title"), "Trial fit title", maximum=110),
        "note": string(data.get("note", ""), "Trial fit note", maximum=2000, required=False),
        "fragments": fragment_refs,
        "candidates": candidates,
        "thresholds": parse_thresholds(data.get("thresholds", {})),
    }


# ---------------------------------------------------------------------------
# Evaluation and persistence
# ---------------------------------------------------------------------------

def load_fragments(store, parsed):
    fragments = {}
    tiles = {}
    for side in SIDES:
        ref = parsed["fragments"][side]
        tile = store.tile(ref["tile_id"])
        plane = tile.get("plane")
        if plane is None or plane["photo_id"] != ref["photo_id"]:
            raise InputError(f"Fragment {side} must use its registered base photograph.")
        if tile.get("width_mm") is None or tile.get("height_mm") is None:
            raise InputError(f"Fragment {side} needs physical width and height in millimetres.")
        tiles[side] = tile
        fragments[side] = Fragment(tile, ref["marks"])
    if parsed["fragments"]["A"]["tile_id"] == parsed["fragments"]["B"]["tile_id"]:
        raise InputError("A trial fit joins two different fragments, not one specimen with itself.")
    return fragments, tiles


def canvas_geometry(fragment_a, fragment_b, evaluations):
    """Compute a common mm canvas that holds every candidate placement."""
    points = [fragment_a.outline_mm()]
    for evaluation in evaluations:
        rotation = np.asarray(evaluation["rotation"], dtype=np.float64)
        translation = np.asarray(evaluation["translation_mm"], dtype=np.float64)
        points.append(fragment_b.outline_mm(rotation, translation))
    all_points = np.vstack(points)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = maximum - minimum
    if np.any(span <= 0):
        raise InputError("The placed fragments do not define a usable base-map area.")
    # One shared mm-per-pixel keeps both fragments at their true physical size.
    mm_per_pixel = float(span.max() / CANVAS_LONGEST_PX)
    px_span = np.maximum(np.ceil(span / mm_per_pixel).astype(int), 32)
    return {
        "origin_mm": [round(float(minimum[0]), 5), round(float(minimum[1]), 5)],
        "canvas_size": [int(px_span[0]), int(px_span[1])],
        "mm_per_pixel": [mm_per_pixel, mm_per_pixel],
    }


def shifted_pose(evaluation, fragment_a, fragment_b, geometry):
    """Re-express a candidate pose relative to the canvas origin in mm."""
    rotation = np.asarray(evaluation["rotation"], dtype=np.float64)
    translation = np.asarray(evaluation["translation_mm"], dtype=np.float64)
    origin = np.asarray(geometry["origin_mm"], dtype=np.float64)
    # Canvas coordinates subtract the common origin; the rigid rotation is
    # unchanged and only the translation picks up the origin shift.
    shifted_translation = translation - rotation @ origin + origin
    return {"rotation": rotation.tolist(), "translation_mm": shifted_translation.tolist()}


def candidate_summary(evaluation):
    return {
        "id": evaluation["id"],
        "label": evaluation["label"],
        "anchors_a": evaluation["anchors_a"],
        "anchors_b": evaluation["anchors_b"],
        "outline_overlap": evaluation["outline_overlap"],
        "broken_edge_total_mm": evaluation["broken_edge_total_mm"],
        "broken_edge_matched_mm": evaluation["broken_edge_matched_mm"],
        "broken_band_mm": evaluation["broken_band_mm"],
        "solid_contact_mm": evaluation["solid_contact_mm"],
        "color_line_count": evaluation["color_line_count"],
        "color_line_offsets_mm": evaluation["color_line_offsets_mm"],
        "color_line_gaps_mm": evaluation["color_line_gaps_mm"],
        "color_line_offset_max_mm": evaluation["color_line_offset_max_mm"],
        "color_line_offset_mean_mm": evaluation["color_line_offset_mean_mm"],
        "passes": evaluation["passes"],
    }


def evaluate_fit(store, parsed):
    fragments, tiles = load_fragments(store, parsed)
    evaluations = [
        evaluate_candidate(fragments["A"], fragments["B"], candidate, parsed["thresholds"])
        for candidate in parsed["candidates"]
    ]
    decision = choose_base_map(evaluations)
    geometry = canvas_geometry(fragments["A"], fragments["B"], evaluations)
    return fragments, tiles, evaluations, decision, geometry


def build_fit_record(store, parsed, *, fit_id=None, created=True):
    fragments, tiles, evaluations, decision, geometry = evaluate_fit(store, parsed)
    candidate_records = []
    for evaluation in evaluations:
        summary = candidate_summary(evaluation)
        # Store every candidate's canvas-space pose so previews and the adopted
        # base map use one identical transform instead of recomputing it.
        summary["canvas_pose"] = shifted_pose(evaluation, fragments["A"], fragments["B"], geometry)
        candidate_records.append(summary)
    record = {
        "id": fit_id or make_id(),
        "title": parsed["title"],
        "note": parsed["note"],
        "thresholds": parsed["thresholds"],
        "canvas_size": geometry["canvas_size"],
        "origin_mm": geometry["origin_mm"],
        "mm_per_pixel": geometry["mm_per_pixel"],
        "fragments": {},
        "candidates": candidate_records,
        "adoption": decision,
        "created_at": None,
        "updated_at": None,
    }
    for side in SIDES:
        ref = parsed["fragments"][side]
        tile = tiles[side]
        fragment = fragments[side]
        member = {
            "tile_id": ref["tile_id"],
            "photo_id": ref["photo_id"],
            "tile_name": tile["name"],
            "marks": ref["marks"],
            "mm_per_pixel": [float(fragment.mm[0]), float(fragment.mm[1])],
        }
        if side == "A":
            member["pose"] = {"rotation": np.eye(2).tolist(),
                              "translation_mm": (-np.asarray(geometry["origin_mm"])).tolist()}
        record["fragments"][side] = member
    if decision is not None:
        winner = next(item for item in evaluations if item["id"] == decision["candidate_id"])
        record["fragments"]["B"]["pose"] = shifted_pose(winner, fragments["A"], fragments["B"], geometry)
        record["base_map_digest"] = stored_preview(base_map_image(store, record), store.blobs)
    else:
        record["fragments"]["B"]["pose"] = None
        record["base_map_digest"] = None
    return record, evaluations, decision


# ---------------------------------------------------------------------------
# Preview rendering (markings on a single registered photo)
# ---------------------------------------------------------------------------

MARK_COLORS = {
    "solid": (120, 231, 185),
    "broken": (246, 139, 96),
    "color": (238, 221, 130),
    "outline": (167, 181, 220),
    "anchor": (244, 245, 225),
}


def marked_photo(store, tile_id, photo_id, marks):
    from .geometry import specimen_frame
    tile = store.tile(tile_id)
    frame = specimen_frame(store.blobs, tile, photo_id)
    image = frame.display()
    from .photos import uint8_image
    canvas = uint8_image(image, frame.valid.astype(np.float32)).convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    stroke = max(1, round(min(width, height) / 320))

    def xy(points):
        return [(round(x * width), round(y * height)) for x, y in points]

    outline = xy(marks["outline"])
    draw.line([*outline, outline[0]], fill=SHADOW, width=stroke + 2, joint="curve")
    draw.line([*outline, outline[0]], fill=MARK_COLORS["outline"], width=stroke, joint="curve")
    for kind, color in (("solid", MARK_COLORS["solid"]), ("broken", MARK_COLORS["broken"])):
        for chain in marks["edges"][kind]:
            points = xy(chain)
            draw.line(points, fill=SHADOW, width=stroke + 3, joint="curve")
            draw.line(points, fill=color, width=stroke + 1, joint="curve")
    for line in marks["color_lines"]:
        points = xy(line)
        draw.line(points, fill=SHADOW, width=stroke + 3)
        draw.line(points, fill=MARK_COLORS["color"], width=stroke + 1)
        x, y = points[1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=MARK_COLORS["color"], outline=SHADOW, width=1)
    label = font(max(12, round(min(width, height) / 42)))
    draw.text((8, 8), tile["name"][:40], font=label, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return canvas


def candidate_preview(store, record, candidate_id):
    """Render a candidate placement using its stored canvas pose."""
    candidate = next((item for item in record["candidates"] if item["id"] == candidate_id), None)
    if candidate is None:
        raise InputError("That candidate placement is not part of this trial fit.")
    if "canvas_pose" not in candidate:
        raise InputError("This saved trial fit predates stored candidate poses; rebuild it to preview.")
    preview = {**record, "fragments": {
        "A": record["fragments"]["A"],
        "B": {**record["fragments"]["B"], "pose": candidate["canvas_pose"]},
    }}
    image = base_map_image(store, preview).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    typeface = font(max(13, round(min(width, height) / 48)))
    status = "passes thresholds" if candidate["passes"] else "fails thresholds"
    draw.text((10, 8), f"{candidate['label']} · {status}", font=typeface, fill=INK,
              stroke_width=2, stroke_fill=SHADOW)
    return image


# ---------------------------------------------------------------------------
# Orchestration used by the web layer
# ---------------------------------------------------------------------------

def create_fit(store, payload):
    parsed = parse_fit_payload(payload)
    record, _, decision = build_fit_record(store, parsed)
    saved = store.save_fit(record)
    if decision is None:
        raise InputError(
            "Neither candidate cleared both thresholds, so no trial-fit base map was adopted. "
            "Adjust the placements or the recorded thresholds."
        )
    return saved, decision


def require_adopted(record):
    if not record.get("adoption") or record.get("base_map_digest") is None:
        raise InputError("This trial fit has no adopted base map. Re-evaluate its candidates first.")
    return record


def create_seam_scan(store, fit_id, payload):
    from .seams import fit_provenance, measure_seam_scan, seam_overlay
    from .store import key_for
    record = require_adopted(store.fit(fit_id))
    data = dictionary(payload, "Fracture line scan")
    options = {
        "title": string(data.get("title"), "Line scan title", maximum=110),
        "start": numeric_pair(data.get("start"), "Line start", lower=0, upper=1),
        "end": numeric_pair(data.get("end"), "Line end", lower=0, upper=1),
        "samples": integer(data.get("samples", 120), "Stations", lower=20, upper=360),
        "band_pixels": scalar(data.get("band_pixels", 5), "Band width px", lower=0.5, upper=32),
    }
    metrics = measure_seam_scan(store, record, options)
    overlay = stored_preview(seam_overlay(store, record, metrics), store.blobs)
    scan = {
        "id": make_id(),
        "fit_id": record["id"],
        "title": options["title"],
        "kind": "seam",
        "label": "Fracture line scan",
        "metrics": metrics,
        "provenance": {
            "fit_id": record["id"],
            "fit_title": record["title"],
            "adopted_candidate_id": record["adoption"]["candidate_id"],
            "canvas_size": record["canvas_size"],
            "mm_per_pixel": record["mm_per_pixel"],
            "photos": fit_provenance(store, record),
        },
        "pictures": {"overlay": overlay, "base_map": record["base_map_digest"]},
        "created_at": None,
    }
    cache = key_for({"fit": record["id"], "options": options, "adoption": record["adoption"]})
    scan["created_at"] = timestamp()
    return store.save_seam_scan(cache, scan)


def station_lookup(store, fit_id, scan_id, station):
    record = require_adopted(store.fit(fit_id))
    scan = store.seam_scan(scan_id)
    if scan["fit_id"] != record["id"]:
        raise InputError("That line scan does not belong to this trial fit.")
    rows = scan["metrics"]["rows"]
    index = integer(station, "Station index", lower=0, upper=len(rows) - 1)
    row = rows[index]
    if row["side"] is None:
        return {"station": index, "side": None,
                "message": "This station sits in the empty seam and has no physical specimen or reading."}
    member = record["fragments"][row["side"]]
    tile = store.tile(member["tile_id"])
    photo = next(item for item in tile["photos"] if item["id"] == row["photo_id"])
    return {
        "station": index,
        "side": row["side"],
        "tile_id": member["tile_id"],
        "tile_name": tile["name"],
        "photo_id": photo["id"],
        "photo_name": photo["name"],
        "local_position": row["local_position"],
        "lightness": row["lightness"],
        "a": row["a"],
        "b": row["b"],
    }
