"""Line scans that cross the fracture between two trial-fitted fragments.

The scan lives on the single adopted trial-fit base map, but every reading is
traced back to the original registered photograph it came from.  Stations
falling in the empty seam (where neither fragment has physical support) are
kept as explicit blank rows: even if the transverse sampling band would still
see fragment colour on either side, a station without a specimen directly
under it is never assigned a colour.
"""
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .art import ACCENT, INK, SHADOW
from .color import linear_lab
from .fitting import apply_pose
from .geometry import measure_line_samples, specimen_frame
from .model import InputError
from .photos import luminance, uint8_image


MIN_VALID_STATIONS = 4


class PlacedFragment:
    """A registered fragment frame plus its rigid pose in the base-map plane."""

    def __init__(self, frame, fragment, rotation, translation, side):
        self.frame = frame
        self.fragment = fragment
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.translation = np.asarray(translation, dtype=np.float64)
        self.side = side
        self.mm = fragment.mm
        self.extent_mm = fragment.extent_mm

    def base_to_local_mm(self, points):
        points = np.asarray(points, dtype=np.float64)
        return apply_pose(points, self.rotation.T, -self.rotation.T @ self.translation)

    def base_to_fraction(self, points):
        # Local millimetres back to normalized photo fractions.
        return self.base_to_local_mm(points) / self.extent_mm

    def local_mm_to_base(self, points):
        return apply_pose(np.asarray(points, dtype=np.float64), self.rotation, self.translation)


def _band_center_coords(extent, start, end, samples):
    width, height = extent
    origin = np.array(start) * [width, height] - 0.5
    finish = np.array(end) * [width, height] - 0.5
    delta = finish - origin
    fractions = np.linspace(0, 1, samples)
    centers = origin[None, :] + fractions[:, None] * delta
    return centers, delta


def support_maps(placed, canvas_size, scale):
    """Rasterize each fragment's outlined support onto the base-map canvas.

    The registered photo can be opaque outside the physical fragment, so the
    marked fragment outline (mapped through the rigid pose) clips ownership.
    Returns ``(map_a, map_b)`` boolean masks with ``mode="constant"`` so the
    empty seam stays empty instead of inheriting an edge pixel by padding.
    """
    from PIL import Image, ImageDraw
    width, height = canvas_size
    yy, xx = np.indices((height, width), dtype=np.float32)
    base_mm = np.stack(((xx + 0.5) * scale[0], (yy + 0.5) * scale[1]), axis=2)
    maps = {}
    for item in placed:
        fractions = item.base_to_fraction(base_mm)
        local_x = fractions[..., 0] * item.frame.extent[0] - 0.5
        local_y = fractions[..., 1] * item.frame.extent[1] - 0.5
        support = ndimage.map_coordinates(
            item.frame.valid.astype(np.float32), (local_y, local_x),
            order=1, mode="constant", cval=0,
        )
        # Outline polygon in base-map pixel coordinates.
        outline_local_mm = item.fragment.to_mm(item.fragment.outline)
        outline_base = item.local_mm_to_base(outline_local_mm)
        outline_px = [(x / scale[0], y / scale[1]) for x, y in outline_base]
        polygon = Image.new("1", (width, height), 0)
        ImageDraw.Draw(polygon).polygon(outline_px, fill=1)
        inside_outline = np.asarray(polygon, dtype=bool)
        maps[item.side] = (support >= 0.999) & inside_outline
    return maps["A"], maps["B"]


def measure_seam_scan(store, fit, options):
    placed = build_placed_fragments(store, fit)
    samples = int(options["samples"])
    start = np.asarray(options["start"], dtype=np.float64)
    end = np.asarray(options["end"], dtype=np.float64)

    extent = fit["canvas_size"]
    width, height = extent
    start_px = start * [width, height]
    end_px = end * [width, height]
    delta_px = end_px - start_px
    line_length_px = float(np.linalg.norm(delta_px))
    if line_length_px < 3:
        raise InputError("Mark two line endpoints more than three base-map pixels apart.")
    scale_mm = np.asarray(fit["mm_per_pixel"], dtype=np.float64)
    line_length_mm = float(np.linalg.norm((end - start) * np.asarray(extent) * scale_mm))
    band_px = float(options["band_pixels"])

    centers_px, _ = _band_center_coords(extent, start, end, samples)
    centers_mm = centers_px * scale_mm
    perpendicular = np.array([-delta_px[1], delta_px[0]]) / line_length_px
    steps = max(int(math.ceil(band_px)), 1)
    transverse = np.linspace(-band_px / 2, band_px / 2, steps) if steps > 1 else np.array([0.0])
    # Transverse sampling offsets expressed in millimetres.
    transverse_mm = transverse * scale_mm[0]

    # Rasterized per-pixel ownership.  A pixel is A, B or empty; the empty seam
    # remains empty even though a wide transverse band would still see colour.
    map_a, map_b = support_maps(placed, extent, scale_mm)
    center_col = np.clip(np.rint(centers_px[:, 0]).astype(int), 0, width - 1)
    center_row = np.clip(np.rint(centers_px[:, 1]).astype(int), 0, height - 1)
    on_a = map_a[center_row, center_col]
    on_b = map_b[center_row, center_col]
    if np.any(on_a & on_b):
        raise InputError(
            "The line crosses a base-map pixel owned by both fragments. The "
            "adopted fit must keep the two outlines from overlapping."
        )
    assignment = [("A" if a else "B" if b else None) for a, b in zip(on_a, on_b)]

    rows = []
    axis_mm = np.linspace(0, line_length_mm, samples)
    for index in range(samples):
        row = {
            "station": index,
            "position": round(float(axis_mm[index]), 5),
            "position_unit": "mm",
            "relative_position": [
                round(float(start[0] + (end[0] - start[0]) * index / (samples - 1)), 6),
                round(float(start[1] + (end[1] - start[1]) * index / (samples - 1)), 6),
            ],
            "side": None,
            "samples": 0,
            "lightness": None,
            "a": None,
            "b": None,
            "intensity": None,
            "photo_id": None,
            "local_position": None,
        }
        side = assignment[index]
        if side is not None:
            owner = next(item for item in placed if item.side == side)
            _fill_station(row, owner, centers_mm[index], perpendicular, transverse_mm, scale_mm)
        rows.append(row)

    valid_rows = [row for row in rows if row["side"] is not None]
    if len(valid_rows) < MIN_VALID_STATIONS:
        raise InputError("The seam line needs at least four stations that land on a fragment.")
    blank = [row["station"] for row in rows if row["side"] is None]
    return {
        "endpoints": [list(start), list(end)],
        "band_pixels": band_px,
        "samples": samples,
        "line_length": round(line_length_mm, 5),
        "distance_unit": "mm",
        "supported_stations": len(valid_rows),
        "blank_stations": blank,
        "rows": rows,
        "features": seam_features(rows, line_length_mm),
    }


def _fill_station(row, placed, center_mm, perpendicular_mm, transverse_mm, scale_mm):
    """Attribute a supported station and read its colour on its own fragment."""
    # Build the transverse band in base-map millimetres, then map into this
    # fragment's registered photo so every reading is traceable to its source.
    band_mm = center_mm[None, :] + transverse_mm[:, None] * perpendicular_mm[None, :]
    fractions = placed.base_to_fraction(band_mm)
    width, height = placed.frame.extent
    local_x = fractions[:, 0] * width - 0.5
    local_y = fractions[:, 1] * height - 0.5
    # One station, with its transverse band across the second axis.
    coords = (local_y[None, :], local_x[None, :])
    domain = placed.frame.valid & ~placed.frame.clipped
    lab, counts = measure_line_samples(linear_lab(placed.frame.linear), domain, coords)
    intensity, _ = measure_line_samples(luminance(placed.frame.linear), domain, coords)
    row["side"] = placed.side
    row["samples"] = int(counts[0])
    if counts[0]:
        row["lightness"] = round(float(lab[0, 0]), 4)
        row["a"] = round(float(lab[0, 1]), 4)
        row["b"] = round(float(lab[0, 2]), 4)
        row["intensity"] = round(float(intensity[0, 0]), 5)
    row["photo_id"] = placed.frame.photo["id"]
    # The selected station maps back to a point on the original registered photo.
    row["local_position"] = [round(value, 6) for value in placed.base_to_fraction(center_mm)]


def seam_features(rows, line_length):
    valid = [row for row in rows if row["lightness"] is not None]
    if not valid:
        return {"first_valid": None, "last_valid": None, "blank_station_runs": blank_runs(rows)}
    lightness = np.asarray([row["lightness"] for row in valid], dtype=np.float64)
    return {
        "first_valid": valid[0]["station"],
        "last_valid": valid[-1]["station"],
        "minimum_lightness": round(float(lightness.min()), 4),
        "maximum_lightness": round(float(lightness.max()), 4),
        "median_lightness": round(float(np.median(lightness)), 4),
        "blank_station_runs": blank_runs(rows),
        "line_length": round(line_length, 5),
    }


def blank_runs(rows):
    runs = []
    start = None
    for row in rows:
        if row["side"] is None and start is None:
            start = row["station"]
        if row["side"] is not None and start is not None:
            runs.append([start, row["station"] - 1])
            start = None
    if start is not None:
        runs.append([start, rows[-1]["station"]])
    return runs


# ---------------------------------------------------------------------------
# Base-map construction and provenance
# ---------------------------------------------------------------------------

def build_placed_fragments(store, fit):
    placed = []
    for member, side in ((fit["fragments"]["A"], "A"), (fit["fragments"]["B"], "B")):
        tile = store.tile(member["tile_id"])
        frame = specimen_frame(store.blobs, tile, member["photo_id"])
        from .fitting import Fragment
        fragment = Fragment(tile, member["marks"])
        rotation = np.asarray(member["pose"]["rotation"], dtype=np.float64)
        translation = np.asarray(member["pose"]["translation_mm"], dtype=np.float64)
        placed.append(PlacedFragment(frame, fragment, rotation, translation, side))
    return placed


def base_map_image(store, fit):
    """Render the adopted fit as one base-map raster with a blank seam."""
    placed = build_placed_fragments(store, fit)
    width, height = fit["canvas_size"]
    scale = np.asarray(fit["mm_per_pixel"], dtype=np.float64)
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    base_mm = np.stack(((xx + 0.5) * scale[0], (yy + 0.5) * scale[1]), axis=2)
    map_a, map_b = support_maps(placed, (width, height), scale)
    ownership = {"A": map_a, "B": map_b}
    for item in placed:
        present = ownership[item.side]
        fractions = item.base_to_fraction(base_mm)
        local_x = fractions[..., 0] * item.frame.extent[0] - 0.5
        local_y = fractions[..., 1] * item.frame.extent[1] - 0.5
        for channel in range(3):
            values = ndimage.map_coordinates(
                item.frame.linear[..., channel], (local_y, local_x), order=1, mode="constant", cval=0,
            )
            canvas[present, channel] = values[present]
        alpha[present] = 1.0
    return uint8_image(canvas, alpha)


def seam_overlay(store, fit, metrics):
    image = base_map_image(store, fit).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    start = tuple(round(v * dim) for v, dim in zip(metrics["endpoints"][0], image.size))
    end = tuple(round(v * dim) for v, dim in zip(metrics["endpoints"][1], image.size))
    stroke = max(2, int(round(metrics["band_pixels"])))
    draw.line((start, end), fill=SHADOW, width=stroke + 3)
    draw.line((start, end), fill=ACCENT, width=stroke)
    for row in metrics["rows"]:
        x = round(row["relative_position"][0] * width)
        y = round(row["relative_position"][1] * height)
        if row["side"] is None:
            color = (236, 121, 125)
        elif row["side"] == "A":
            color = (120, 200, 231)
        else:
            color = (238, 200, 120)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=SHADOW, width=1)
    for index, point in enumerate((start, end)):
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=INK, outline=SHADOW, width=2)
        draw.text((point[0] + 7, point[1] - 7), "AB"[index], fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def fit_provenance(store, fit):
    photos = []
    for side in ("A", "B"):
        member = fit["fragments"][side]
        tile = store.tile(member["tile_id"])
        photo = next(item for item in tile["photos"] if item["id"] == member["photo_id"])
        photos.append({
            "side": side,
            "tile_id": member["tile_id"],
            "tile_name": tile["name"],
            "photo_id": photo["id"],
            "display_name": photo["name"],
            "role": photo["role"],
            "original_digest": photo["source_digest"],
            "working_digest": photo["working_digest"],
            "plane_size": tile["plane"]["size"],
            "physical_mm": [tile["width_mm"], tile["height_mm"]],
            "mm_per_pixel": list(member["mm_per_pixel"]),
        })
    return photos
