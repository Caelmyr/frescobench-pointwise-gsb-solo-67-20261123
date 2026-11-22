"""Synthetic two-fragment trial-fit training material.

The two opaque panels share one painted colour line that reaches the fracture.
Candidate 1 closes the fracture; candidate 2 shifts the second fragment so the
colour line misses, giving a passing and a failing placement against the same
two conservator thresholds.
"""
import numpy as np
from PIL import Image, ImageDraw

from .fits import build_fit_record
from .photos import png_bytes, store_capture
from .store import make_id

PLANE_PX_PER_MM = 10
TRIAL_KEY = "frescobench.trial.v1"
FRAGMENT_W_MM = 40.0
FRAGMENT_H_MM = 30.0
GAP_MM = 2.0
LINE_Y_MM = 15.0
LINE_WIDTH_PX = 3
LINE_COLOR = (64, 72, 122)
BASE_COLORS = ((196, 178, 142), (178, 158, 128))


def _panel(color, line_y_px, seed):
    width = int(FRAGMENT_W_MM * PLANE_PX_PER_MM)
    height = int(FRAGMENT_H_MM * PLANE_PX_PER_MM)
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 5, (height, width, 1))
    array = np.clip(np.asarray(color, dtype=np.float32)[None, None, :] + grain, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    draw = ImageDraw.Draw(image)
    # A dark painted line ending at the fracture-facing (right) edge.
    draw.line((0, line_y_px, width - 1, line_y_px), fill=LINE_COLOR, width=LINE_WIDTH_PX)
    return image


def _tile(store, name, wall, color, seed):
    width_px = int(FRAGMENT_W_MM * PLANE_PX_PER_MM)
    line_y_px = int(LINE_Y_MM * PLANE_PX_PER_MM)
    image = _panel(color, line_y_px, seed)
    photo = store_capture(
        store.blobs, png_bytes(image), "diffuse",
        f"{name.lower().replace(' ', '-')}.png",
        "Synthetic registered fresco fragment with a millimetre plane.",
        make_id(), synthetic=True,
    )
    return {
        "name": name,
        "batch": "Trial tray T1",
        "body": "Lime plaster",
        "glaze": "Mineral line on limewash",
        "firing": wall,
        "notes": "Synthetic trial-fit fragment with a millimetre-registered survey plane.",
        "width_mm": FRAGMENT_W_MM,
        "height_mm": FRAGMENT_H_MM,
        "synthetic": True,
        "photos": [photo],
        "plane": {"photo_id": photo["id"], "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                  "size": [width_px, int(FRAGMENT_H_MM * PLANE_PX_PER_MM)]},
        "regions": [],
        "registrations": {},
    }


def _marks():
    # The outline stays just inside the traced edges.  For fragment A the
    # fracture-facing right side sits just inside the broken edge.
    outline = [[0.03, 0.07], [0.95, 0.07], [0.95, 0.93], [0.03, 0.93]]
    return {
        "outline": outline,
        "edges": {
            # Solid (intact, un-broken) edges.  For fragment A the outer solid
            # edge is the left edge; the facing top/bottom edges are traced
            # separately so contact length only counts facing solid edges.
            "solid": [
                [[0.02, 0.06], [0.02, 0.94]],
                [[0.04, 0.04], [0.96, 0.04]],
                [[0.96, 0.96], [0.04, 0.96]],
            ],
            # Broken edge: the fracture-facing vertical edge.  It is a straight
            # survey trace here so the synthetic seam keeps a uniform width.
            "broken": [
                [[0.977, 0.06], [0.977, 0.94]],
            ],
        },
        "color_lines": [
            # Interior end first, fracture-facing end second.
            [[0.12, LINE_Y_MM / FRAGMENT_H_MM], [0.96, LINE_Y_MM / FRAGMENT_H_MM]],
        ],
    }


def _marks_b():
    marks = _marks()
    # Mirror horizontally so the broken edge faces left toward fragment A.
    def mirror(points):
        return [[1.0 - x, y] for x, y in points]
    outline = mirror(marks["outline"])
    # Mirroring reverses winding; restore clockwise order so the polygon stays simple.
    from frescobench.fitting import _polygon_orientation
    if _polygon_orientation(np.asarray(outline)) < 0:
        outline = outline[::-1]
    solid = [mirror(chain) for chain in marks["edges"]["solid"]]
    broken = [mirror(marks["edges"]["broken"][0])[::-1]]
    # After mirroring the fracture is on the left.  Point order must remain
    # [interior end (right), fracture-facing end (left)].
    mirrored_line = mirror(marks["color_lines"][0])
    color_lines = [[mirrored_line[0], mirrored_line[1]]]
    return {"outline": outline, "edges": {"solid": solid, "broken": broken}, "color_lines": color_lines}


def _payload(store, tile_a, tile_b):
    # In the assembled wall the right broken edge of A nearly touches the left
    # broken edge of B, separated by the empty GAP_MM seam.
    a_broken_x = 0.977
    b_broken_x = 0.023
    # Equal inset from each broken edge so rigid alignment brings the two
    # fracture faces together; the conservator's seam_gap_mm tolerance then
    # accepts the small physical clearance when evaluating the match.
    inset_fraction = 1.0 / FRAGMENT_W_MM
    anchor_a_x = a_broken_x - inset_fraction
    anchor_b_x = b_broken_x + inset_fraction
    anchor_a_close = [[anchor_a_x, 0.12], [anchor_a_x, 0.86]]
    anchor_b_close = [[anchor_b_x, 0.12], [anchor_b_x, 0.86]]
    # Candidate 2 shifts B downward so the colour line misses the fracture.
    shift_mm = 3.6
    shift_fraction = shift_mm / FRAGMENT_H_MM
    anchor_b_shift = [[anchor_b_x, 0.12 + shift_fraction], [anchor_b_x, 0.86 + shift_fraction]]
    return {
        "title": "Trial fit · two blue-line fragments",
        "note": "Candidate 1 closes the fracture; candidate 2 shifts fragment B down by 3.6 mm.",
        "fragments": {
            "A": {"tile_id": tile_a["id"], "photo_id": tile_a["plane"]["photo_id"], "marks": _marks()},
            "B": {"tile_id": tile_b["id"], "photo_id": tile_b["plane"]["photo_id"], "marks": _marks_b()},
        },
        "candidates": [
            {"id": "close", "label": "Candidate 1 · fracture closed",
             "anchors_a": anchor_a_close, "anchors_b": anchor_b_close},
            {"id": "shifted", "label": "Candidate 2 · shifted 3.6 mm down",
             "anchors_a": anchor_a_close, "anchors_b": anchor_b_shift},
        ],
        # Conservator-supplied minimums: at least 20 mm of matchable broken
        # edge, no more than 2.5 mm of colour-line offset at the break and a
        # 2 mm empty seam between the two fracture faces.
        "thresholds": {
            "minimum_matched_mm": 20.0,
            "maximum_line_offset_mm": 2.5,
            "seam_gap_mm": GAP_MM,
        },
    }


def load_trial_pair(store):
    existing = store.seeded_tiles(TRIAL_KEY)
    if existing:
        tile_a, tile_b = existing
        fit = next((item for item in store.list_fits()
                    if item["fragments"]["A"]["tile_id"] == tile_a["id"]), None)
        return tile_a, tile_b, fit
    tile_a = store.insert_seed_tile(_tile(store, "Fragment T1-A", "North wall", BASE_COLORS[0], 71), TRIAL_KEY)
    tile_b = store.insert_seed_tile(_tile(store, "Fragment T1-B", "North wall", BASE_COLORS[1], 83), TRIAL_KEY)
    from .fits import parse_fit_payload
    parsed = parse_fit_payload(_payload(store, tile_a, tile_b))
    record, _, decision = build_fit_record(store, parsed)
    fit = store.save_fit(record)
    return tile_a, tile_b, fit
