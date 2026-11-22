"""Carry reviewed spot identities from one surface-observation run to the next.

Both runs are registered on the same measured plane. A pair is eligible only
when each spot's center lies on unclipped opaque capture support in the other
run and when the user's center-offset and footprint-overlap limits are met.
The relation is one-to-one in both directions.

Eligible pairs are chosen in this order:

1. keep as many continuation pairs as possible;
2. keep as many earlier spots whose earlier review was ``confirm`` as possible;
3. make the summed footprint agreement as high as possible;
4. make the summed centre offset as small as possible.

Footprint agreement is the intersection over union of the two rasterized
footprints, recorded to 0.00001 and ranked to 0.001 per pair. Centre offset is
the distance in registered-plane pixels, recorded to 0.001 px and ranked as a
fraction of the user's limit to 0.001 per pair. Integer weights make these four
rules lexicographic in one assignment; the remaining tie is left to the
solver's deterministic ordering in reported observation order.

An earlier spot that is not continued stays in the archive with either
``not_photographed`` or ``no_qualifying_continuation``; it is never recorded as
removed. The later run's own review decisions and the earlier run's review
decisions are never rewritten by continuation. A manual reassignment records a
history entry and keeps both runs' reviews intact.
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from .geometry import specimen_frame
from .model import InputError, dictionary, identity, model_version, scalar, string, timestamp

CENTER_OFFSET_UNIT = "registered plane pixels"
FOOTPRINT_MEASURE = "intersection over union of the two outlined footprints"
CAPTURE_RULE = (
    "A position is photographed in a run when it lies on that run's unclipped opaque "
    "capture support. Continuation never needs an inspection region to be drawn over it."
)
RANKING_RULE = (
    "Most continuation pairs first; then most earlier confirmed spots continued; then greatest "
    "summed footprint overlap; then smallest summed centre offset. One-to-one in both directions."
)
OUTCOMES = {
    "continued": "Continued",
    "not_photographed": "Not photographed in the other view",
    "no_qualifying_continuation": "No qualifying continuation",
}
LATER_OUTCOMES = {
    "continued": "Continues an earlier spot",
    "not_photographed": "Only in this view",
    "no_qualifying_continuation": "No qualifying earlier spot",
}

# The four rules are applied lexicographically through exact integer weights in
# a single assignment. Per pair:
#   count step       1e13  (above the largest possible sum of the rules below)
#   confirmed step   1e11  (above the largest possible overlap+offset sum)
#   overlap          2e5 per thousandth (at most 160 * 1000 * 2e5)
#   offset           1 per thousandth of the remaining offset allowance
# A move of one pair therefore always outranks every lower-rule gain, and so on.
# The largest total is about 1.6e15, below 2^53, so float64 holds every weight
# exactly and the ranking is never decided by rounding. Runs report at most
# 160 observations each.
COUNT_STEP = 10**13
CONFIRM_STEP = 10**11
OVERLAP_STEP = 2 * 10**5
OVERLAP_UNITS = 1000
OFFSET_UNITS = 1000
INELIGIBLE_WEIGHT = 1e16
TRANSFORM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Footprint:
    left: int
    top: int
    mask: np.ndarray

    @property
    def right(self):
        return self.left + self.mask.shape[1]

    @property
    def bottom(self):
        return self.top + self.mask.shape[0]

    @property
    def area(self):
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class RunSupport:
    extent: tuple
    captured: np.ndarray


def transform_matches(stored, current):
    if len(stored) != len(current):
        return False
    return all(abs(float(a) - float(b)) <= TRANSFORM_TOLERANCE for a, b in zip(stored, current))


def run_support(store, tile, record):
    provenance = record["provenance"]["photo"]
    frame = specimen_frame(store.blobs, tile, provenance["id"])
    stored_extent = tuple(record["support"]["registered_image_extent"])
    if tuple(frame.extent) != stored_extent:
        raise InputError(
            "The measured plane has changed since that run. Continuation needs runs made on the same plane."
        )
    if not transform_matches(provenance["plane_to_capture"], frame.transform):
        raise InputError(
            "The capture alignment has changed since that run. Re-register both captures on the same plane first."
        )
    return RunSupport(tuple(frame.extent), frame.valid & ~frame.clipped)


def rasterize(observation, extent):
    width, height = extent
    bounds = observation["bounds"]
    left = max(0, int(math.floor(bounds[0] * width)))
    top = max(0, int(math.floor(bounds[1] * height)))
    right = min(width, int(math.ceil(bounds[2] * width)))
    bottom = min(height, int(math.ceil(bounds[3] * height)))
    if right <= left or bottom <= top:
        empty = np.zeros((1, 1), dtype=bool)
        return Footprint(0, 0, empty)
    mask = Image.new("1", (right - left, bottom - top))
    points = [
        (float(x) * width - 0.5 - left, float(y) * height - 0.5 - top)
        for x, y in observation["outline"]
    ]
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=1)
    else:
        ImageDraw.Draw(mask).rectangle((0, 0, right - left - 1, bottom - top - 1), fill=1)
    return Footprint(left, top, np.asarray(mask, dtype=bool))


def overlap(first, second):
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    first_crop = first.mask[top - first.top:bottom - first.top, left - first.left:right - first.left]
    second_crop = second.mask[top - second.top:bottom - second.top, left - second.left:right - second.left]
    intersection = int(np.count_nonzero(first_crop & second_crop))
    union = first.area + second.area - intersection
    return 0.0 if union <= 0 else intersection / union


def overlap_upper_bound(first, second):
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    largest = max(first.area, second.area)
    if largest <= 0:
        return 0.0
    return ((right - left) * (bottom - top)) / largest


def captured_at(captured, center, extent):
    width, height = extent
    col = min(width - 1, max(0, int(math.floor(center[0] * width))))
    row = min(height - 1, max(0, int(math.floor(center[1] * height))))
    return bool(captured[row, col])


def center_offset(first, second, extent):
    return math.hypot(
        (first["normalized_center"][0] - second["normalized_center"][0]) * extent[0],
        (first["normalized_center"][1] - second["normalized_center"][1]) * extent[1],
    )


def previous_decision(record, key):
    mark = record.get("decisions", {}).get(key, {})
    return mark.get("decision", "review"), mark.get("notes", ""), mark.get("updated_at")


def index_of(record):
    return {entry["key"]: index for index, entry in enumerate(record["metrics"]["observations"])}


def eligible_pairs(store, tile, earlier, later, earlier_support, later_support, offset_limit, overlap_limit):
    if offset_limit <= 0:
        raise InputError("The center-offset limit must be greater than zero.")
    if not 0 <= overlap_limit <= 1:
        raise InputError("The footprint-overlap limit must be between 0 and 1.")
    extent = earlier_support.extent
    earlier_order = index_of(earlier)
    later_order = index_of(later)
    pairs = []
    later_shapes = {}
    later_eligible = []
    for observation in later["metrics"]["observations"]:
        if captured_at(earlier_support.captured, observation["normalized_center"], extent):
            later_shapes[observation["key"]] = rasterize(observation, extent)
            later_eligible.append(observation)
    for earlier_observation in earlier["metrics"]["observations"]:
        if not captured_at(later_support.captured, earlier_observation["normalized_center"], extent):
            continue
        earlier_shape = rasterize(earlier_observation, extent)
        for later_observation in later_eligible:
            distance = center_offset(earlier_observation, later_observation, extent)
            if distance > offset_limit:
                continue
            later_shape = later_shapes[later_observation["key"]]
            if overlap_limit > 0 and overlap_upper_bound(earlier_shape, later_shape) < overlap_limit:
                continue
            agreement = overlap(earlier_shape, later_shape)
            if agreement + 1e-12 < overlap_limit:
                continue
            status, notes, reviewed_at = previous_decision(earlier, earlier_observation["key"])
            pairs.append({
                "earlier": earlier_observation["key"],
                "later": later_observation["key"],
                "earlier_index": earlier_order[earlier_observation["key"]],
                "later_index": later_order[later_observation["key"]],
                "center_offset_px": round(distance, 3),
                "footprint_overlap": round(agreement, 5),
                "previous_decision": status,
                "previous_notes": notes,
                "previous_reviewed_at": reviewed_at,
            })
    pairs.sort(key=lambda item: (item["earlier_index"], item["later_index"]))
    return pairs


def pair_value(pair, offset_limit):
    fraction = min(1.0, max(0.0, pair["center_offset_px"] / offset_limit))
    value = COUNT_STEP
    if pair["previous_decision"] == "confirm":
        value += CONFIRM_STEP
    value += int(round(pair["footprint_overlap"] * OVERLAP_UNITS)) * OVERLAP_STEP
    value += int(round((1 - fraction) * OFFSET_UNITS))
    return value


def pair_cost(pair, offset_limit):
    return -pair_value(pair, offset_limit)


def choose_pairs(pairs, earlier_count, later_count, offset_limit):
    """Best one-to-one relation under the ranking rule."""
    if not pairs:
        return []
    costs = np.full((earlier_count, later_count), INELIGIBLE_WEIGHT, dtype=np.float64)
    lookup = {}
    for pair in pairs:
        costs[pair["earlier_index"], pair["later_index"]] = pair_cost(pair, offset_limit)
        lookup[(pair["earlier_index"], pair["later_index"])] = pair
    rows, cols = linear_sum_assignment(costs)
    chosen = [
        lookup[(int(row), int(col))]
        for row, col in zip(rows, cols)
        if (int(row), int(col)) in lookup
    ]
    return sorted(chosen, key=lambda item: (item["earlier_index"], item["later_index"]))


def link_from_pair(pair):
    return {
        "earlier": pair["earlier"],
        "later": pair["later"],
        "center_offset_px": pair["center_offset_px"],
        "footprint_overlap": pair["footprint_overlap"],
    }


def archive(earlier, later, links, earlier_support, later_support, cleared=()):
    extent = earlier_support.extent
    set_aside = set(cleared)
    by_earlier = {item["earlier"]: item for item in links}
    by_later = {item["later"]: item for item in links}
    earlier_rows = []
    for observation in earlier["metrics"]["observations"]:
        key = observation["key"]
        status, notes, reviewed_at = previous_decision(earlier, key)
        if key in by_earlier:
            pair = by_earlier[key]
            outcome = "continued"
        elif not captured_at(later_support.captured, observation["normalized_center"], extent):
            pair = None
            outcome = "not_photographed"
        else:
            pair = None
            outcome = "no_qualifying_continuation"
        earlier_rows.append({
            "key": key,
            "type": observation["type"],
            "center": observation["normalized_center"],
            "previous_decision": status,
            "previous_notes": notes,
            "previous_reviewed_at": reviewed_at,
            "outcome": outcome,
            "outcome_label": OUTCOMES[outcome],
            "set_aside": key in set_aside and outcome != "continued",
            "later_key": pair["later"] if pair else None,
            "center_offset_px": pair["center_offset_px"] if pair else None,
            "footprint_overlap": pair["footprint_overlap"] if pair else None,
        })
    later_rows = []
    for observation in later["metrics"]["observations"]:
        key = observation["key"]
        status, notes, reviewed_at = previous_decision(later, key)
        if key in by_later:
            pair = by_later[key]
            outcome = "continued"
        elif not captured_at(earlier_support.captured, observation["normalized_center"], extent):
            pair = None
            outcome = "not_photographed"
        else:
            pair = None
            outcome = "no_qualifying_continuation"
        later_rows.append({
            "key": key,
            "type": observation["type"],
            "center": observation["normalized_center"],
            "this_decision": status,
            "this_notes": notes,
            "this_reviewed_at": reviewed_at,
            "outcome": outcome,
            "outcome_label": LATER_OUTCOMES[outcome],
            "earlier_key": pair["earlier"] if pair else None,
            "center_offset_px": pair["center_offset_px"] if pair else None,
            "footprint_overlap": pair["footprint_overlap"] if pair else None,
        })
    return earlier_rows, later_rows


def summarize(earlier_rows, later_rows):
    return {
        "earlier_total": len(earlier_rows),
        "later_total": len(later_rows),
        "continued": sum(row["outcome"] == "continued" for row in earlier_rows),
        "earlier_not_photographed": sum(row["outcome"] == "not_photographed" for row in earlier_rows),
        "earlier_no_qualifying_continuation": sum(row["outcome"] == "no_qualifying_continuation" for row in earlier_rows),
        "later_not_photographed": sum(row["outcome"] == "not_photographed" for row in later_rows),
        "later_no_qualifying_continuation": sum(row["outcome"] == "no_qualifying_continuation" for row in later_rows),
        "confirmed_continued": sum(
            row["outcome"] == "continued" and row["previous_decision"] == "confirm" for row in earlier_rows
        ),
        "confirmed_archived": sum(
            row["previous_decision"] == "confirm" and row["outcome"] != "continued" for row in earlier_rows
        ),
    }


def rules(offset_limit, overlap_limit):
    return {
        "capture": CAPTURE_RULE,
        "footprint": FOOTPRINT_MEASURE,
        "ranking": RANKING_RULE,
        "offset_unit": CENTER_OFFSET_UNIT,
        "limits": {"center_offset_px": offset_limit, "footprint_overlap": overlap_limit},
    }


def validate_runs(earlier, later):
    for record, label in ((earlier, "The earlier run"), (later, "This run")):
        if record["kind"] != "defects":
            raise InputError(f"{label} is not a Surface observations measurement.")
    if earlier["tile_id"] != later["tile_id"]:
        raise InputError("Both runs must belong to the same specimen.")
    if earlier["id"] == later["id"]:
        raise InputError("Choose a different earlier run.")
    if earlier["provenance"]["photo"]["id"] == later["provenance"]["photo"]["id"]:
        raise InputError("Choose a run made from a different photograph.")


def validate_source(store, earlier, later):
    """Keep a chain of runs acyclic. A later run may only descend from an older one."""
    seen = {later["id"]}
    cursor = earlier
    while cursor is not None:
        if cursor["id"] in seen:
            raise InputError("These runs would form a loop of continuations. Choose an earlier run that does not already descend from this one.")
        seen.add(cursor["id"])
        following = cursor.get("continuation")
        cursor = store.result(following["earlier_id"]) if following else None


def supports(store, earlier, later):
    validate_runs(earlier, later)
    validate_source(store, earlier, later)
    tile = store.tile(later["tile_id"])
    earlier_support = run_support(store, tile, earlier)
    later_support = run_support(store, tile, later)
    if earlier_support.extent != later_support.extent:
        raise InputError("Both runs must be registered on the same measured plane size.")
    return tile, earlier_support, later_support


def refresh(store, earlier, later, continuation):
    """Rebuild the archive and counts from the saved relation without changing it."""
    _, earlier_support, later_support = supports(store, earlier, later)
    earlier_rows, later_rows = archive(
        earlier, later, continuation["links"], earlier_support, later_support, continuation.get("cleared", [])
    )
    return {
        **continuation,
        "archive": {"earlier": earlier_rows, "later": later_rows},
        "summary": summarize(earlier_rows, later_rows),
    }


def link_options(store, earlier, later, continuation):
    """Eligible later spots for each earlier spot under the saved limits."""
    tile, earlier_support, later_support = supports(store, earlier, later)
    pairs = eligible_pairs(
        store, tile, earlier, later,
        earlier_support, later_support,
        continuation["limits"]["center_offset_px"], continuation["limits"]["footprint_overlap"],
    )
    earlier_order = index_of(earlier)
    later_order = index_of(later)
    return [
        {
            "earlier": pair["earlier"],
            "later": pair["later"],
            "earlier_index": earlier_order[pair["earlier"]],
            "later_index": later_order[pair["later"]],
            "center_offset_px": pair["center_offset_px"],
            "footprint_overlap": pair["footprint_overlap"],
            "previous_decision": pair["previous_decision"],
        }
        for pair in pairs
    ]


def apply_choice(store, earlier, later, continuation, earlier_key, later_key):
    """Move exactly one earlier spot, then refill only spots left without a partner.

    Other saved relations are never rewritten. A spot that this edit leaves open
    receives a partner only from free pairs, so a manual change cannot take away
    a continuation another open spot could have kept. A move onto a later spot
    that another earlier spot already continues is refused; the other relation
    must be cleared first so the saved relation is never silently ambiguous.
    """
    options = link_options(store, earlier, later, continuation)
    earlier_index = index_of(earlier)
    if earlier_key not in earlier_index:
        raise InputError("That earlier spot is not in the earlier run.")
    if later_key is not None:
        if later_key not in index_of(later):
            raise InputError("That later spot is not in this run.")
        allowed = next(
            (item for item in options if item["earlier"] == earlier_key and item["later"] == later_key),
            None,
        )
        if allowed is None:
            raise InputError("That pair is not photographed in both views or is outside the saved limits.")
        taken = next(
            (item for item in continuation["links"] if item["later"] == later_key and item["earlier"] != earlier_key),
            None,
        )
        if taken is not None:
            raise InputError(f"That later spot already continues earlier spot {taken['earlier']}.")
    links = [item for item in continuation["links"] if item["earlier"] != earlier_key]
    claimed_earlier = {item["earlier"] for item in links}
    claimed_later = {item["later"] for item in links}
    cleared = set(continuation.get("cleared", []))
    if later_key is None:
        cleared.add(earlier_key)
    else:
        cleared.discard(earlier_key)
        links.append({
            "earlier": earlier_key,
            "later": later_key,
            "center_offset_px": next(
                item["center_offset_px"] for item in options
                if item["earlier"] == earlier_key and item["later"] == later_key
            ),
            "footprint_overlap": next(
                item["footprint_overlap"] for item in options
                if item["earlier"] == earlier_key and item["later"] == later_key
            ),
        })
        claimed_earlier.add(earlier_key)
        claimed_later.add(later_key)
    # A cleared spot stays out even though refilling cannot change the number of
    # continuations by more than one; the clear is the user's explicit choice.
    free_pairs = [
        item for item in options
        if item["earlier"] not in claimed_earlier
        and item["later"] not in claimed_later
        and item["earlier"] not in cleared
    ]
    extras = choose_pairs(
        free_pairs, len(earlier["metrics"]["observations"]),
        len(later["metrics"]["observations"]), continuation["limits"]["center_offset_px"],
    )
    for pair in extras:
        links.append({
            "earlier": pair["earlier"],
            "later": pair["later"],
            "center_offset_px": pair["center_offset_px"],
            "footprint_overlap": pair["footprint_overlap"],
        })
    links.sort(key=lambda item: earlier_index[item["earlier"]])
    return {**continuation, "links": links, "cleared": sorted(cleared)}


def successors_of(records, record_id):
    return [
        record for record in records
        if record["id"] != record_id and record.get("continuation", {}).get("earlier_id") == record_id
    ]


def earlier_options(store, record):
    extent = tuple(record["support"]["registered_image_extent"])
    photo_id = record["provenance"]["photo"]["id"]
    options = []
    records = store.results(record["tile_id"])
    for candidate in records:
        if candidate["kind"] != "defects" or candidate["id"] == record["id"]:
            continue
        if candidate["provenance"]["photo"]["id"] == photo_id:
            continue
        if tuple(candidate["support"]["registered_image_extent"]) != extent:
            continue
        decisions = candidate.get("decisions", {})
        options.append({
            "id": candidate["id"],
            "label": candidate["label"],
            "created_at": candidate["created_at"],
            "source_version": candidate["source_version"],
            "photo_name": candidate["provenance"]["photo"]["display_name"],
            "photo_role": candidate["provenance"]["photo"]["role"],
            "confirmed_count": sum(item.get("decision") == "confirm" for item in decisions.values()),
            "dismissed_count": sum(item.get("decision") == "dismiss" for item in decisions.values()),
            "observation_count": len(candidate["metrics"]["observations"]),
            "has_continuation": bool(candidate.get("continuation")),
        })
    options.sort(key=lambda item: item["created_at"], reverse=True)
    return options


def view(store, record):
    continuation = record.get("continuation")
    records = store.results(record["tile_id"])
    later = successors_of(records, record["id"])
    payload = {
        "measurement_id": record["id"],
        "continuation": continuation,
        "earlier_options": earlier_options(store, record),
        "successors": [
            {
                "id": item["id"],
                "created_at": item["created_at"],
                "photo_role": item["provenance"]["photo"]["role"],
                "photo_name": item["provenance"]["photo"]["display_name"],
                "continued": item["continuation"]["summary"]["continued"],
                "earlier_total": item["continuation"]["summary"]["earlier_total"],
            }
            for item in later
        ],
        "candidate_pairs": [],
        "warning": None,
    }
    if not continuation:
        return payload
    try:
        earlier = store.result(continuation["earlier_id"])
        refreshed = refresh(store, earlier, record, continuation)
        payload["continuation"] = refreshed
        payload["candidate_pairs"] = link_options(store, earlier, record, refreshed)
        payload["earlier_measurement"] = {
            "id": earlier["id"],
            "label": earlier["label"],
            "created_at": earlier["created_at"],
            "source_version": earlier["source_version"],
            "photo": {
                "id": earlier["provenance"]["photo"]["id"],
                "name": earlier["provenance"]["photo"]["display_name"],
                "role": earlier["provenance"]["photo"]["role"],
                "overlay_url": f"/api/results/{earlier['id']}/image/overlay",
                "source_url": f"/api/results/{earlier['id']}/image/source",
            },
            "decisions": earlier.get("decisions", {}),
            "observations": earlier["metrics"]["observations"],
        }
    except InputError as error:
        payload["warning"] = str(error)
    return payload


def parse_limits(payload):
    data = dictionary(payload, "Continuation limits")
    offset = scalar(data.get("center_offset_px"), "Center-offset limit", lower=0.01, upper=100000)
    overlap = scalar(data.get("footprint_overlap"), "Footprint-overlap limit", lower=0, upper=1)
    return offset, overlap


def resolve(store, record, payload):
    """Existing relation for this run, or a fresh suggestion from the chosen earlier run.

    ``resuggest: true`` rebuilds the automatic relation from the entered limits
    and discards manual moves (they remain in the history). Without it, a saved
    relation is returned untouched so a page refresh never overwrites a choice.
    """
    data = dictionary(payload, "Continuation")
    continuation = record.get("continuation")
    earlier_id = data.get("earlier_id")
    resuggest = data.get("resuggest") is True
    if earlier_id is not None:
        identity(earlier_id, "Earlier measurement ID")
    if continuation:
        if earlier_id is not None and continuation["earlier_id"] != earlier_id:
            raise InputError("This run already continues a different earlier run.")
        existing = store.result(continuation["earlier_id"])
        if not resuggest:
            return existing, refresh(store, existing, record, continuation)
        offset, overlap = parse_limits(payload)
        suggestion = suggest(store, existing, record, offset, overlap)
        history = list(continuation.get("history", []))
        if any(entry.get("action") in ("reassigned", "unlinked") for entry in history):
            history.append({
                "changed_at": timestamp(),
                "action": "resuggested",
                "earlier": None,
                "from": None,
                "to": None,
                "reason": "Automatic relation rebuilt from the entered limits; manual moves kept in this history.",
            })
        suggestion["history"] = history
        suggestion["cleared"] = []
        suggestion["created_at"] = continuation.get("created_at", suggestion["created_at"])
        return existing, suggestion
    if earlier_id is None:
        raise InputError("Choose the earlier run to continue.")
    offset, overlap = parse_limits(payload)
    return store.result(earlier_id), suggest(store, store.result(earlier_id), record, offset, overlap)


def suggest(store, earlier, record, offset, overlap):
    _, earlier_support, later_support = supports(store, earlier, record)
    pairs = eligible_pairs(
        store, store.tile(record["tile_id"]), earlier, record,
        earlier_support, later_support, offset, overlap,
    )
    chosen = choose_pairs(
        pairs, len(earlier["metrics"]["observations"]), len(record["metrics"]["observations"]), offset
    )
    earlier_rows, later_rows = archive(
        earlier, record, [link_from_pair(pair) for pair in chosen], earlier_support, later_support
    )
    created = timestamp()
    return {
        "earlier_id": earlier["id"],
        "earlier_label": earlier["label"],
        "earlier_created_at": earlier["created_at"],
        "earlier_source_version": earlier["source_version"],
        "earlier_photo": {
            "id": earlier["provenance"]["photo"]["id"],
            "name": earlier["provenance"]["photo"]["display_name"],
            "role": earlier["provenance"]["photo"]["role"],
        },
        "limits": {"center_offset_px": offset, "footprint_overlap": overlap},
        "rules": rules(offset, overlap),
        "links": [link_from_pair(pair) for pair in chosen],
        "cleared": [],
        "history": [],
        "archive": {"earlier": earlier_rows, "later": later_rows},
        "summary": summarize(earlier_rows, later_rows),
        "created_at": created,
        "updated_at": created,
        "candidate_count": len(pairs),
    }


def save(store, record, payload):
    expected = model_version(payload)
    store.require_version(record, expected)
    earlier, suggestion = resolve(store, record, payload)
    return store.save_continuation(
        record["id"],
        expected,
        suggestion,
        "start",
        f"Continued {suggestion['summary']['continued']} of {suggestion['summary']['earlier_total']} earlier observations",
    )


def change_link(store, record, payload):
    data = dictionary(payload, "Observation continuation")
    expected = model_version(data)
    continuation = record.get("continuation")
    if not continuation:
        raise InputError("Start a continuation for this run before changing a spot relationship.")
    earlier_key = string(data.get("earlier"), "Earlier observation ID", maximum=8)
    later_value = data.get("later")
    later_key = string(later_value, "Later observation ID", maximum=8) if later_value is not None else None
    earlier = store.result(continuation["earlier_id"])
    previous = next((item for item in continuation["links"] if item["earlier"] == earlier_key), None)
    previous_later = previous["later"] if previous else None
    if previous_later == later_key:
        return record
    store.require_version(record, expected)
    updated = apply_choice(store, earlier, record, continuation, earlier_key, later_key)
    reason = string(data.get("reason"), "Continuation note", maximum=240, required=False)
    history = list(continuation.get("history", []))
    history.append({
        "changed_at": timestamp(),
        "action": "reassigned" if previous_later and later_key else "unlinked" if previous_later else "linked",
        "earlier": earlier_key,
        "from": previous_later,
        "to": later_key,
        "reason": reason,
    })
    updated["history"] = history
    if later_key is not None:
        for item in updated["links"]:
            if item["earlier"] == earlier_key:
                item["history"] = [
                    entry for entry in history if entry["earlier"] == earlier_key
                ]
    updated = refresh(store, earlier, record, updated)
    return store.save_continuation(
        record["id"],
        expected,
        updated,
        "link" if later_key else "unlink",
        f"{earlier_key} → {later_key or 'no later observation'}",
    )

