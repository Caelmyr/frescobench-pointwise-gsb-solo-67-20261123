from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry import DEFAULT_CORNERS, canvas_dimensions, homography
from .photos import linear_to_srgb, png_bytes, srgb_to_linear
from .store import make_id
from .photos import store_capture


DEMO_KEY = "frescobench.training.v1"
PLANE_SIZE = (620, 472)
PHOTO_SIZE = (766, 598)
DEMO_ROLES = ("diffuse", "parallel", "cross", "rake-left", "rake-right")
POSES = {
    "diffuse": [[0.092, 0.086], [0.911, 0.112], [0.894, 0.912], [0.081, 0.889]],
    "parallel": [[0.122, 0.107], [0.910, 0.084], [0.925, 0.871], [0.101, 0.916]],
    "cross": [[0.079, 0.101], [0.908, 0.119], [0.877, 0.916], [0.081, 0.909]],
    "rake-left": [[0.089, 0.094], [0.894, 0.088], [0.902, 0.905], [0.064, 0.911]],
    "rake-right": [[0.094, 0.095], [0.913, 0.094], [0.926, 0.906], [0.112, 0.893]],
}


@dataclass(frozen=True)
class DemoSpecimen:
    seed: int
    name: str
    body: str
    glaze: str
    firing: str
    batch: str
    colors: tuple


SERIES = (
    DemoSpecimen(173, "Ultramarine · W12", "Lime plaster", "Blue mineral pigment", "North wall", "Tray A",
                ((27, 56, 75), (42, 92, 117), (81, 133, 147))),
    DemoSpecimen(284, "Ochre red · W03", "Fine lime mortar", "Red ochre wash", "Window reveal", "Tray B",
                ((82, 34, 27), (134, 57, 40), (189, 101, 64))),
    DemoSpecimen(391, "Lime grey · W06", "Gypsum plaster", "Neutral limewash", "West wall", "Tray A",
                ((56, 62, 64), (108, 124, 126), (163, 176, 166))),
    DemoSpecimen(475, "Earth green · W11", "Lime plaster", "Green earth pigment", "Vault", "Tray C",
                ((39, 65, 63), (67, 108, 95), (114, 151, 118))),
)


def random_field(generator, size, y_space, x_space):
    height, width = size
    rows, cols = int(np.ceil(height / y_space)) + 4, int(np.ceil(width / x_space)) + 4
    coarse = generator.normal(size=(rows, cols)).astype(np.float32)
    y, x = np.indices(size, dtype=np.float32)
    field = ndimage.map_coordinates(coarse, (y / y_space + 1, x / x_space + 1), order=3)
    return field.astype(np.float32)


@dataclass(frozen=True)
class ProofField:
    diffuse: np.ndarray
    height: np.ndarray
    ink_fraction: np.ndarray
    keyline_mask: np.ndarray
    roughness: np.ndarray
    u: np.ndarray
    v: np.ndarray


def surface_palette(colors, values):
    linear_colors = srgb_to_linear(np.array(colors, dtype=np.float32) / 255)
    stops = (-1.2, 0.05, 1.25)
    return np.stack(
        [np.interp(values, stops, linear_colors[:, channel]) for channel in range(3)],
        axis=2,
    ).astype(np.float32)


def neutral_card(diffuse, roughness, x, y):
    sample = (x >= 0.07) & (x <= 0.165) & (y >= 0.844) & (y <= 0.961)
    border = (x >= 0.063) & (x <= 0.172) & (y >= 0.835) & (y <= 0.968) & ~sample
    diffuse[sample] = np.array([0.194, 0.172, 0.155])
    diffuse[border] *= 0.70
    roughness[sample] = 0.76


def folio_marks(x, y, rng):
    lower_tag = (x > .70) & (x < .94) & (y > .785) & (y < .943)
    lines = ((np.floor(y * 235).astype(int) % 6) == 0) & lower_tag & (x > .717) & (x < .919)
    scale = (x > .48) & (x < .66) & (y > .886) & (y < .938)
    dark_ticks = (np.floor((x - .48) * 225).astype(int) % 2 == 0) & scale
    folio_number = (x > .763) & (x < .875) & (y > .810) & (y < .818)
    tape_1 = (x > .250) & (x < .276) & (y > .372) & (y < .517)
    tape_2 = (x > .613) & (x < .629) & (y > .475) & (y < .635)
    sheet = np.broadcast_to(np.array([.68, .62, .49], dtype=np.float32), (*x.shape, 3)).copy()
    paper = rng.normal(0, .014, x.shape).astype(np.float32)
    warm = (.027 * np.sin(4*x+3*y) + .042*x) * (1 - y)
    sheet += (paper + warm)[..., None]
    sheet[lower_tag | scale] += [.12, .13, .14]
    sheet[lines | dark_ticks | folio_number] = [.14, .15, .12]
    return sheet, tape_1 | tape_2


def bake_field(specimen):
    width, height = PLANE_SIZE
    shape = (height, width)
    generator = np.random.default_rng(specimen.seed)
    y, x = np.indices(shape, dtype=np.float32)
    y /= height
    x /= width
    broad = random_field(generator, shape, 47, 68)
    mottling = random_field(generator, shape, 11, 19)
    grain = generator.normal(size=shape).astype(np.float32)
    sheet, tape = folio_marks(x, y, generator)
    tooth = random_field(generator, shape, 5.5, 6.1)
    chipped_top = .170 + .019 * np.cos(14 * x) + .041 * np.exp(-((x - .45) / .11) ** 2)
    broken_bottom = .793 - .034 * np.cos(9 * x) - .033 * np.exp(-((x - .30) / .13) ** 2)
    shifted_left = .119 + .029 * np.sin(12 * y) + .037 * (1 - y)
    shifted_right = .894 - .17 * np.maximum((y - .68) / .28, 0) + .013 * np.sin(31 * y)
    edge_noise = .0035 * tooth
    fragment = ((x > shifted_left + edge_noise) & (x < shifted_right + edge_noise)
                & (y > chipped_top + edge_noise) & (y < broken_bottom + edge_noise))
    brush = np.sin(240 * x + 55 * y + .65 * mottling)
    pigment_value = .45 * broad + .25 * mottling + .21 * tooth + .11 * brush + .069 * grain
    paint = surface_palette(specimen.colors, pigment_value)
    relief = (.013 * broad + .038 * tooth + .010 * brush) * fragment
    roughness = np.where(fragment, np.clip(.53 + .14 * tooth, .31, .79), .78)

    plaster = np.array([.47, .39, .28], dtype=np.float32)
    worn = np.array([.39, .20, .07], dtype=np.float32)
    edge_margins = fragment & ((x < shifted_left + .019) | (x > shifted_right - .015)
                               | (y < chipped_top + .015) | (y > broken_bottom - .017))
    paint[edge_margins] = plaster + grain[edge_margins, None] * .025
    curved_band = np.abs(y - (.27 + .30 * (x - .36) ** 2 + .024 * np.sin(9 * x))) < .020
    motif = curved_band & fragment & (x > .20) & (x < .77)
    pigment_trace = np.clip(.57 + .16 * mottling[motif], .12, .78)
    paint[motif] += pigment_trace[:, None] * (worn - paint[motif])
    relief[motif] += .009

    # A marked loss strip has one intact right end and asymmetric bare margins.
    centerline = .439 + .0028 * np.sin(17 * x)
    left_halo = .006 + .025 * np.exp(-((x - .43) / .12) ** 2)
    right_halo = .005 + .039 * np.exp(-((x - .58) / .11) ** 2)
    issued = fragment & (x > .243) & (x < .788)
    rough_boundary = .0048 * tooth + .0046 * np.sin(57 * x)
    exposed = issued & (y > centerline - left_halo - rough_boundary) & (y < centerline + right_halo + rough_boundary)
    scribe = issued & (np.abs(y - centerline) < .0029)
    chipping = np.clip(.34 + .23 * mottling + .19 * tooth, 0, 1)
    exposed_color = plaster + chipping[..., None] * (worn - plaster) + grain[..., None] * .017
    paint[exposed] = exposed_color[exposed]
    paint[scribe] = np.clip(plaster + grain[scribe, None] * .037, 0, 1)
    relief[exposed] -= .023
    roughness[exposed] = .68

    control = fragment & (x > .709) & (x < .854) & (y > .642) & (y < .735)
    brushing = .024 * np.sin(690 * y) + .016 * grain
    paint[control] = plaster + brushing[control, None]
    roughness[control] = .72

    for index in range(42):
        cx, cy = generator.uniform((.166, .237), (.852, .730))
        rx, ry = generator.uniform(.0025, .010, 2)
        radius = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        pinhole = (radius < 1 + .10 * tooth) & fragment & ~control & ~exposed
        paint[pinhole] *= np.clip(.08 + .16 * radius[pinhole, None], .05, .26)
        relief[pinhole] -= .08

    mineral = np.array([.68, .675, .645], dtype=np.float32)
    for cx, cy, radius_x, radius_y in ((.64, .337, .045, .027), (.691, .359, .020, .012), (.50, .67, .028, .034)):
        footprint = ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2
        bloom = fragment & (footprint < 1 + .30 * tooth + .23 * mottling)
        crystals = .025 * grain[bloom] + .068 * tooth[bloom]
        paint[bloom] = np.clip(mineral + crystals[:, None], .34, .91)
        roughness[bloom] = .87
        relief[bloom] += .036

    marks = np.zeros(shape, dtype=bool)
    for cx in (.080, .918):
        for cy in (.146, .81):
            marks |= ((abs(x - cx) < .014) & (abs(y - cy) < .0024)) | ((abs(x - cx) < .0019) & (abs(y - cy) < .016))
    diffuse = np.where(fragment[..., None], paint, sheet * [.78, .91, 1.01])
    diffuse[marks] = [.045, .06, .062]
    neutral_card(diffuse, roughness, x, y)
    specular = np.where(fragment, .52 - .30 * exposed, .04)
    return ProofField(np.clip(diffuse, 0, 1), relief, specular,
                      (tape | marks | scribe).astype(np.float32), roughness, x, y)


def gloss_reflection(field, role):
    x, y = field.u, field.v
    hotspot = np.exp(-(((x - 0.64 + 0.17 * y) / 0.21) ** 2 + ((y - 0.38) / 0.25) ** 2))
    elongated = np.exp(-(((x - 0.64 - 0.38 * (y - 0.3)) / 0.075) ** 2 + ((y - 0.49) / 0.31) ** 2))
    low = (1 - field.roughness) * field.ink_fraction * (0.5 + 0.5 * np.sin(8 * x + 13 * y) ** 2)
    intensity = 0.079 * low if role == "diffuse" else 0.34 * low + 0.24 * hotspot + 0.12 * elongated
    if role == "cross":
        intensity = (0.009 + hotspot * 0.018) * low
    return intensity * (1 - field.keyline_mask * 0.58)


def light_field(field, role):
    x, y = field.u, field.v
    dy, dx = np.gradient(0.92 * field.height)
    broad = 0.93 + 0.13 * (1 - y) - 0.08 * (x - 0.5) ** 2
    if role in ("rake-left", "rake-right"):
        direction = 1 if role == "rake-left" else -1
        side = 0.91 + direction * 0.13 * (0.5 - x)
        shading = np.clip(0.66 + (dx * direction - dy * 0.23) * 3.9, 0.26, 1.32)
        rgb = field.diffuse * (broad * side * shading)[:, :, None]
    else:
        side = np.clip(0.98 + dx * 0.38 - dy * 0.31, 0.64, 1.2)
        reflection = gloss_reflection(field, role)
        exposure = 0.88 if role == "cross" else 1.09 if role == "parallel" else 1.0
        rgb = field.diffuse * (broad * side * exposure)[:, :, None]
        rgb += reflection[:, :, None] * np.array([0.69, 0.79, 0.82], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def desk_background(width, height, seed):
    generator = np.random.default_rng(seed + 500)
    noise = random_field(generator, (height, width), 21, 30) * 0.018
    y, x = np.indices((height, width), dtype=np.float32)
    grad = 0.12 + 0.19 * np.exp(-((x / width - 0.24) ** 2 + (y / height - 0.10) ** 2) * 2.4)
    grain = generator.normal(size=(height, width)).astype(np.float32) * 0.008
    neutral = np.maximum(grad + grain + noise, 0.035)
    return neutral[:, :, None] * np.array([0.50, 0.57, 0.58], dtype=np.float32)


def surface_alpha(x, y):
    horizontal = np.abs(x - 0.5) - 0.461
    vertical = np.abs(y - 0.5) - 0.452
    corner = np.hypot(np.maximum(horizontal, 0), np.maximum(vertical, 0))
    nearest = np.minimum(np.maximum(horizontal, vertical), 0)
    sdf = corner + nearest - 0.035
    return np.clip(0.5 - sdf * 410, 0, 1)


def warp_to_photo(array, corners, photo_size, seed):
    width, height = photo_size
    photo_to_tile = np.linalg.inv(homography(DEFAULT_CORNERS, corners))
    photo_y, photo_x = np.indices((height, width), dtype=np.float32)
    photo_y = (photo_y + 0.5) / height
    photo_x = (photo_x + 0.5) / width
    denominator = photo_to_tile[2, 0] * photo_x + photo_to_tile[2, 1] * photo_y + photo_to_tile[2, 2]
    tile_u = (photo_to_tile[0, 0] * photo_x + photo_to_tile[0, 1] * photo_y + photo_to_tile[0, 2]) / denominator
    tile_v = (photo_to_tile[1, 0] * photo_x + photo_to_tile[1, 1] * photo_y + photo_to_tile[1, 2]) / denominator
    tile_h, tile_w = array.shape[:2]
    coords = (tile_v * tile_h - 0.5, tile_u * tile_w - 0.5)
    tile_image = np.stack(
        [ndimage.map_coordinates(array[:, :, component], coords, order=1, mode="nearest") for component in range(3)],
        axis=2,
    )
    alpha = surface_alpha(tile_u, tile_v)
    shadow = ndimage.shift(ndimage.gaussian_filter(alpha, 7), (9, 5), order=1, mode="constant")
    stage = desk_background(width, height, seed)
    stage *= (1 - 0.61 * shadow)[:, :, None]
    rim = np.clip(alpha - ndimage.minimum_filter(alpha, size=8), 0, 1)
    tile_image *= 1 - 0.24 * rim[:, :, None]
    mixed = stage * (1 - alpha[:, :, None]) + tile_image * alpha[:, :, None]
    return linear_to_srgb(mixed)


def render_photo(field, role, specimen):
    linear = light_field(field, role)
    image = warp_to_photo(linear, POSES[role], PHOTO_SIZE, specimen.seed)
    array = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def rectangle(name, purpose, left, top, right, bottom):
    return {
        "id": make_id(), "name": name, "purpose": purpose,
        "vertices": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def training_regions():
    painted_face = {
        "id": make_id(), "name": "Painted face", "purpose": "inspection",
        "vertices": [[.184, .225], [.33, .228], [.46, .260], [.58, .224], [.72, .200],
                     [.854, .238], [.866, .390], [.860, .623], [.817, .730],
                     [.64, .740], [.40, .745], [.177, .729], [.143, .566], [.139, .441]],
    }
    return [
        painted_face,
        rectangle("Intact pigment", "glazed", 0.32, 0.555, 0.38, 0.602),
        rectangle("Bare plaster control", "bare", 0.736, 0.658, 0.816, 0.708),
        rectangle("Neutral card", "neutral", 0.086, 0.858, 0.15, 0.943),
        rectangle("Image scale", "ignore", 0.478, 0.883, 0.663, 0.941),
        rectangle("Marked loss strip", "inspection", .223, .353, .812, .517),
    ]


def training_specimen(store, specimen):
    field = bake_field(specimen)
    photos = []
    for role in DEMO_ROLES:
        image = render_photo(field, role, specimen)
        photo = store_capture(
            store.blobs, png_bytes(image), role,
            f"{specimen.name.lower().replace(' ', '-')}-{role}.png",
            "Synthetic wall-paint fragment with a loss margin, bare plaster control and pale deposits.",
            make_id(),
            synthetic=True,
        )
        photos.append(photo)
    base = photos[0]
    size = canvas_dimensions(base, POSES["diffuse"], 112, 84)
    return {
        "name": specimen.name,
        "batch": specimen.batch,
        "body": specimen.body,
        "glaze": specimen.glaze,
        "firing": specimen.firing,
        "notes": "Synthetic wall-paint fragment. Intact pigment, exposed plaster and pale deposits are visible in registered views.",
        "width_mm": 112,
        "height_mm": 84,
        "synthetic": True,
        "photos": photos,
        "plane": {"photo_id": base["id"], "corners": POSES["diffuse"], "size": list(size)},
        "regions": training_regions(),
        "registrations": {
            photo["id"]: {"base_photo_id": base["id"], "corners": POSES[photo["role"]], "support_percent": 100}
            for photo in photos[1:]
        },
    }


def load_training(store):
    existing = store.seeded_tiles(DEMO_KEY)
    if existing:
        return existing, False
    rows = [training_specimen(store, specimen) for specimen in SERIES]
    return store.seed_tiles(DEMO_KEY, rows), True


@lru_cache(maxsize=1)
def preview_photo():
    specimen = SERIES[0]
    rendered = render_photo(bake_field(specimen), "diffuse", specimen)
    return png_bytes(rendered)
