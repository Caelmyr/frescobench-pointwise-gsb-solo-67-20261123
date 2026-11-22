"""Trial-fit geometry for two scattered wall-painting fragments.

A fit keeps both fragments at their registered physical size.  The first
fragment (``A``) is fixed; the second (``B``) is moved by a rigid transform
(rotation + translation only, never a scale or shear).  Every distance used
for the thresholds and ranking is expressed in millimetres derived from the
two registered survey planes.

The conservation decision represented here is intentionally rule based:

* a candidate first has to clear two thresholds supplied by the conservator
  (a minimum matchable broken-edge length and a maximum colour-line offset at
  the fracture);
* among the candidates that clear both thresholds the candidate with the
  longest physically contacting intact (solid) edge wins;
* if two candidates contact the same length of solid edge the one whose two
  colour lines line up more closely wins.

Only one candidate is ever adopted as the single trial-fit base map.
"""
import math

import numpy as np

from .model import InputError


# A contact is "physical" when the two solid edges are within a fraction of a
# registered pixel of each other after placement.
CONTACT_TOLERANCE_PX = 0.75
BROKEN_BAND_PX = 2.5
RIGID_SCALE_TOLERANCE = 0.01
MIN_CHAIN_POINTS = 2
MAX_CHAIN_POINTS = 200
SAMPLES_PER_MM = 6.0


# ---------------------------------------------------------------------------
# Generic polyline helpers (coordinates are plain numpy arrays)
# ---------------------------------------------------------------------------

def _coords(points, label, *, minimum, maximum, unit_bounded):
    if not isinstance(points, (list, tuple, np.ndarray)) or not minimum <= len(points) <= maximum:
        raise InputError(f"{label} needs between {minimum} and {maximum} marked points.")
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (len(points), 2) or not np.all(np.isfinite(array)):
        raise InputError(f"{label} points must each be an X, Y coordinate pair.")
    if unit_bounded and (np.any(array < 0) or np.any(array > 1)):
        raise InputError(f"{label} points must lie inside the registered photo (0 to 1).")
    return array


def _as_points(points, label, *, minimum=MIN_CHAIN_POINTS, maximum=MAX_CHAIN_POINTS):
    return _coords(points, label, minimum=minimum, maximum=maximum, unit_bounded=True)


def chain_length(points):
    points = np.asarray(points, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def resample_polyline(points, spacing):
    """Return vertices spaced roughly ``spacing`` apart along the polyline."""
    points = np.asarray(points, dtype=np.float64)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total = float(np.sum(segment_lengths))
    if total <= 1e-12:
        return points[:1]
    count = max(2, int(math.ceil(total / max(spacing, 1e-9))) + 1)
    targets = np.linspace(0.0, total, count)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sampled = np.empty((count, 2), dtype=np.float64)
    segment = 0
    for index, target in enumerate(targets):
        while segment < len(segment_lengths) - 1 and cumulative[segment + 1] < target:
            segment += 1
        base = cumulative[segment]
        span = segment_lengths[segment]
        ratio = 0.0 if span <= 1e-12 else (target - base) / span
        sampled[index] = points[segment] + ratio * deltas[segment]
    return sampled


def point_to_segments(point, starts, vectors, lengths):
    """Minimum Euclidean distance from one point to a set of segments."""
    vectors = vectors[lengths > 1e-12]
    starts = starts[lengths > 1e-12]
    lengths = lengths[lengths > 1e-12]
    if not len(lengths):
        return math.inf
    ratios = np.clip(np.sum((point - starts) * vectors, axis=1) / (lengths ** 2), 0.0, 1.0)
    nearest = starts + ratios[:, None] * vectors
    return float(np.min(np.linalg.norm(nearest - point, axis=1)))


def segments_for(points):
    points = np.asarray(points, dtype=np.float64)
    starts = points[:-1]
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    return starts, vectors, lengths


def distance_to_polyline(point, polyline):
    starts, vectors, lengths = segments_for(polyline)
    return point_to_segments(np.asarray(point, dtype=np.float64), starts, vectors, lengths)


# ---------------------------------------------------------------------------
# Polygon overlap (the two fragment outlines may never intersect)
# ---------------------------------------------------------------------------

def _polygon_orientation(points):
    points = np.asarray(points, dtype=np.float64)
    total = 0.0
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        total += first[0] * second[1] - second[0] * first[1]
    return total


def _segments_cross(a, b, c, d):
    p = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    q = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
    r = (d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0])
    s = (d[0] - c[0]) * (b[1] - c[1]) - (d[1] - c[1]) * (b[0] - c[0])
    # A strict sign change means the segments cross through each other's
    # interiors. Endpoint/edge contact (fracture faces sharing a boundary) is
    # not an interior overlap and must not reject a touching trial fit.
    if ((p > 0 and q < 0) or (p < 0 and q > 0)) and ((r > 0 and s < 0) or (r < 0 and s > 0)):
        return True
    return False


def _point_in_polygon(point, polygon):
    inside = False
    x, y = point
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        xi, yi = first
        xj, yj = second
        if (yi > y) != (yj > y):
            crossing = xi + (xj - xi) * (y - yi) / (yj - yi)
            if x < crossing:
                inside = not inside
    return inside


def polygons_overlap(first, second):
    """True when two simple polygons share interior area or cross edges.

    Boundary-only contact (touching fracture faces or corners) is not an
    overlap, because a trial fit brings the broken edges up to one another
    without either fragment occupying the other's interior area.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if abs(_polygon_orientation(first)) < 1e-12 or abs(_polygon_orientation(second)) < 1e-12:
        raise InputError("Each fragment outline needs a non-zero area.")
    for i, a in enumerate(first):
        b = first[(i + 1) % len(first)]
        for j, c in enumerate(second):
            d = second[(j + 1) % len(second)]
            if _segments_cross(a, b, c, d):
                return True
    return _point_in_polygon(second[0], first) or _point_in_polygon(first[0], second)


# ---------------------------------------------------------------------------
# Rigid motion: B is rotated + translated only, preserving physical size
# ---------------------------------------------------------------------------

def rigid_placement(anchors_a, anchors_b):
    """Build a rigid transform mapping B's anchors onto A's anchors.

    Returns a ``(rotation 2x2, translation 2)`` pair in the same coordinate
    units as the supplied points.  Two non-coincident anchor pairs are
    required so the rotation is fully determined.
    """
    anchors_a = _coords(anchors_a, "Anchor pair on fragment A", minimum=2, maximum=2, unit_bounded=False)
    anchors_b = _coords(anchors_b, "Anchor pair on fragment B", minimum=2, maximum=2, unit_bounded=False)
    if chain_length(anchors_a) < 1e-6 or chain_length(anchors_b) < 1e-6:
        raise InputError("The two anchors on each fragment must be distinct points.")
    centroid_a = anchors_a.mean(axis=0)
    centroid_b = anchors_b.mean(axis=0)
    centered_a = anchors_a - centroid_a
    centered_b = anchors_b - centroid_b
    covariance = centered_b.T @ centered_a
    unit_u, _, unit_vt = np.linalg.svd(covariance)
    rotation = unit_u @ unit_vt
    if np.linalg.det(rotation) < 0:
        unit_vt[-1, :] *= -1
        rotation = unit_u @ unit_vt
    translation = centroid_a - rotation @ centroid_b
    scale_a = np.linalg.norm(anchors_a[1] - anchors_a[0])
    scale_b = np.linalg.norm(anchors_b[1] - anchors_b[0])
    if abs(scale_a - scale_b) / scale_b > RIGID_SCALE_TOLERANCE:
        raise InputError(
            "The anchor spacing differs by more than 1%. A trial fit cannot "
            "rescale a fragment; align matching physical points instead."
        )
    return rotation, translation


def apply_pose(points, rotation, translation):
    points = np.asarray(points, dtype=np.float64)
    return points @ rotation.T + translation


# ---------------------------------------------------------------------------
# Marks and candidate normalisation
# ---------------------------------------------------------------------------

EDGE_KINDS = ("solid", "broken")


def normalize_chains(raw, label):
    if not isinstance(raw, dict):
        raise InputError(f"{label} must map edge kind to marked polylines.")
    chains = {}
    for kind in EDGE_KINDS:
        entries = raw.get(kind)
        if not isinstance(entries, list) or not entries:
            raise InputError(f"{label} needs at least one {kind}-edge polyline.")
        chains[kind] = [_as_points(polyline, f"{label} {kind} edge {index + 1}") for index, polyline in enumerate(entries)]
    return chains


def normalize_color_lines(raw, label):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 12:
        raise InputError(f"{label} needs between one and twelve colour lines.")
    lines = []
    for index, entry in enumerate(raw, 1):
        points = _as_points(entry, f"{label} colour line {index}", minimum=2, maximum=2)
        if np.linalg.norm(points[1] - points[0]) < 1e-9:
            raise InputError(f"{label} colour line {index} needs a distinct fracture-facing end.")
        lines.append(points)
    return lines


def normalize_outline(raw, label):
    return _as_points(raw, label, minimum=3, maximum=MAX_CHAIN_POINTS)


class Fragment:
    """A registered fragment with its mm scale and conservator marks."""

    def __init__(self, tile, marks):
        self.tile_id = tile["id"]
        self.name = tile.get("name", self.tile_id)
        plane = tile.get("plane")
        if plane is None:
            raise InputError(f"Fragment {self.name} has no registered survey plane.")
        width_mm = tile.get("width_mm")
        height_mm = tile.get("height_mm")
        if width_mm is None or height_mm is None:
            raise InputError(f"Fragment {self.name} needs width and height in millimetres for a trial fit.")
        plane_width, plane_height = plane["size"]
        self.size_px = (int(plane_width), int(plane_height))
        # Millimetres per registered pixel, and the full physical extent.
        self.mm = np.asarray([width_mm / plane_width, height_mm / plane_height], dtype=np.float64)
        self.extent_mm = np.asarray([width_mm, height_mm], dtype=np.float64)
        self.photo_id = plane["photo_id"]
        self.outline = normalize_outline(marks["outline"], f"{self.name} outline")
        self.chains = normalize_chains(marks["edges"], f"{self.name} edges")
        self.color_lines = normalize_color_lines(marks.get("color_lines", []), f"{self.name}")

    def to_mm(self, points):
        # Normalized photo fractions map to millimetres via the full extent.
        return np.asarray(points, dtype=np.float64) * self.extent_mm

    def outline_mm(self, rotation=None, translation=None):
        outline = self.to_mm(self.outline)
        if rotation is not None:
            outline = apply_pose(outline, rotation, translation)
        return outline

    def chains_mm(self, kind, rotation=None, translation=None):
        result = []
        for chain in self.chains[kind]:
            chain_mm = self.to_mm(chain)
            if rotation is not None:
                chain_mm = apply_pose(chain_mm, rotation, translation)
            result.append(chain_mm)
        return result

    def color_lines_mm(self, rotation=None, translation=None):
        result = []
        for line in self.color_lines:
            line_mm = self.to_mm(line)
            if rotation is not None:
                line_mm = apply_pose(line_mm, rotation, translation)
            result.append(line_mm)
        return result


def _sampled_chain_length_mm(chains, spacing_mm):
    samples = [resample_polyline(chain, spacing_mm) for chain in chains]
    lengths = [chain_length(sample) for sample in samples]
    return samples, sum(lengths)


def evaluate_candidate(fragment_a, fragment_b, candidate, thresholds):
    """Evaluate one proposed rigid placement of B against fixed A."""
    # Anchors arrive as normalized photo fractions; convert each side to its
    # own millimetre plane before solving the rigid motion.
    anchors_a_mm = fragment_a.to_mm(candidate["anchors_a"])
    anchors_b_mm = fragment_b.to_mm(candidate["anchors_b"])
    rotation, translation = rigid_placement(anchors_a_mm, anchors_b_mm)

    outline_a = fragment_a.outline_mm()
    outline_b = fragment_b.outline_mm(rotation, translation)
    overlaps = polygons_overlap(outline_a, outline_b)

    spacing = float(np.minimum.reduce([fragment_a.mm.min(), fragment_b.mm.min()])) / SAMPLES_PER_MM
    broken_a = fragment_a.chains_mm("broken")
    broken_b = fragment_b.chains_mm("broken", rotation, translation)
    solid_a = fragment_a.chains_mm("solid")
    solid_b = fragment_b.chains_mm("solid", rotation, translation)

    # The two fracture faces are separated by the conservator-set seam gap, so
    # a match is measured within that gap plus a small sampling tolerance
    # rather than requiring the broken edges to coincide at zero distance.
    seam_gap = float(thresholds["seam_gap_mm"])
    px_mm = float(min(fragment_a.mm.min(), fragment_b.mm.min()))
    broken_band = seam_gap + BROKEN_BAND_PX * px_mm
    contact_band = seam_gap + CONTACT_TOLERANCE_PX * px_mm

    broken_targets = np.vstack([resample_polyline(chain, spacing) for chain in broken_b])
    broken_segments = [segments_for(chain) for chain in broken_a]

    within = np.zeros(len(broken_targets), dtype=bool)
    for index, point in enumerate(broken_targets):
        distance = min(point_to_segments(point, *seg) for seg in broken_segments)
        within[index] = distance <= broken_band
    # Length is taken from B's sampled broken edge, so it is independent of the
    # sampling density and never exceeds B's own broken-edge length.
    broken_matched_mm = 0.0
    for chain in broken_b:
        count = max(2, int(math.ceil(chain_length(chain) / spacing)) + 1)
        broken_matched_mm += chain_length(resample_polyline(chain, spacing))
    if len(broken_targets):
        broken_matched_mm *= float(np.count_nonzero(within)) / len(broken_targets)
    broken_total_mm = sum(chain_length(chain) for chain in broken_b)

    # Physically contacting solid (intact) edge: walk B's solid chains and keep
    # the arc length whose samples sit within the seam contact band of an A
    # solid edge.  Only facing solid edges can be this close, so distant or
    # outer solid edges contribute nothing.  Length is integrated per segment
    # rather than counted by samples so the metric is in true millimetres.
    solid_segments_a = [segments_for(chain) for chain in solid_a]
    solid_contact_mm = 0.0
    for chain in solid_b:
        dense = resample_polyline(chain, spacing)
        if len(dense) < 2:
            continue
        near = np.array([
            min(point_to_segments(point, *seg) for seg in solid_segments_a) <= contact_band
            for point in dense
        ], dtype=bool)
        step = chain_length(chain) / (len(dense) - 1)
        solid_contact_mm += float(np.count_nonzero(near)) * step

    lines_a = fragment_a.color_lines_mm()
    lines_b = fragment_b.color_lines_mm(rotation, translation)
    paired = min(len(lines_a), len(lines_b))
    if paired == 0:
        raise InputError("Mark at least one visible colour line reaching the break on each fragment.")
    # Colour lines are stored as [interior end, fracture-facing end].  The
    # registration offset is the lateral miss between the two fracture-facing
    # ends measured perpendicular to B's line: it captures how far the painted
    # line is shifted along the break, ignoring the intentional empty seam.
    offsets = []
    gap_lengths = []
    for index in range(paired):
        end_a = lines_a[index][1]
        end_b = lines_b[index][1]
        direction = lines_b[index][1] - lines_b[index][0]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise InputError("A colour line needs two distinct ends to measure its offset.")
        unit = direction / norm
        delta = end_a - end_b
        lateral = abs(float(delta[0] * unit[1] - delta[1] * unit[0]))
        offsets.append(lateral)
        gap_lengths.append(float(abs(np.dot(delta, unit))))
    line_offset_max = max(offsets)
    line_offset_mean = float(np.mean(offsets))

    minimum_match = float(thresholds["minimum_matched_mm"])
    maximum_offset = float(thresholds["maximum_line_offset_mm"])
    passes = (
        not overlaps
        and broken_matched_mm >= minimum_match
        and line_offset_max <= maximum_offset
    )
    return {
        "id": candidate["id"],
        "label": candidate.get("label", candidate["id"]),
        "anchors_a": np.asarray(candidate["anchors_a"], dtype=np.float64).tolist(),
        "anchors_b": np.asarray(candidate["anchors_b"], dtype=np.float64).tolist(),
        "rotation": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "outline_overlap": bool(overlaps),
        "broken_edge_total_mm": round(broken_total_mm, 4),
        "broken_edge_matched_mm": round(broken_matched_mm, 4),
        "broken_band_mm": round(broken_band, 4),
        "seam_gap_mm": round(seam_gap, 4),
        "solid_contact_mm": round(solid_contact_mm, 4),
        "color_line_count": paired,
        "color_line_offsets_mm": [round(value, 4) for value in offsets],
        "color_line_gaps_mm": [round(value, 4) for value in gap_lengths],
        "color_line_offset_max_mm": round(line_offset_max, 4),
        "color_line_offset_mean_mm": round(line_offset_mean, 4),
        "thresholds": {
            "minimum_matched_mm": minimum_match,
            "maximum_line_offset_mm": maximum_offset,
            "seam_gap_mm": seam_gap,
        },
        "passes": bool(passes),
    }


def choose_base_map(evaluations):
    """Apply the documented ranking and return the single adopted candidate."""
    passing = [item for item in evaluations if item["passes"]]
    if not passing:
        return None
    # Primary key: longest physically contacting solid edge.
    # Secondary, documented tiebreaker: closest colour-line registration.
    passing.sort(key=lambda item: (-item["solid_contact_mm"], item["color_line_offset_max_mm"],
                                   item["color_line_offset_mean_mm"], item["id"]))
    winner = passing[0]
    tied = [
        item["id"] for item in passing
        if item["solid_contact_mm"] == winner["solid_contact_mm"]
    ]
    decision = "longest-solid-contact"
    if len(tied) > 1:
        decision = "solid-contact-tie-broken-by-colour-line-offset"
    return {
        "candidate_id": winner["id"],
        "rule": decision,
        "tied_on_solid_contact": tied,
        "passing_candidate_ids": [item["id"] for item in passing],
    }
