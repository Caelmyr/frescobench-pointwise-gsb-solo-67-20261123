// Trial-fit workbench: evaluate two candidate fragment placements, adopt one
// base map, then draw fracture line scans whose stations keep their source.
import {q, api, request, icon, e, btn, badge, empty, img, stat, table, n, date, field, textField, numField, selectField, option, plot} from "./widgets.js";

// state.page is owned by main.js; "fits" and "fit-detail" both render here.
export const fitState = {page: "fits", fits: [], specimens: [], fit: null, scans: [], scan: null, token: 0};

let toast = () => {};
let openModal = null;
let closeModal = null;

export function registerFitShell({notify, modal, dismiss}) {
  toast = notify;
  openModal = modal;
  closeModal = dismiss;
}

async function loadSpecimens() {
  fitState.specimens = (await api("/api/specimens")).items;
}

export async function loadFits() {
  fitState.fits = (await api("/api/fits")).items;
}

export async function openFitList() {
  const token = ++fitState.token;
  await Promise.all([loadFits(), loadSpecimens()]);
  if (token !== fitState.token) return;
  fitState.page = "fits";
  fitState.fit = null;
  renderFitPage();
}

export async function openFit(fitId) {
  const token = ++fitState.token;
  const [fit, scans] = await Promise.all([api(`/api/fits/${fitId}`), api(`/api/fits/${fitId}/scans`)]);
  if (token !== fitState.token) return;
  fitState.fit = fit;
  fitState.scans = scans.items;
  fitState.scan = null;
  fitState.page = "fit-detail";
  renderFitPage();
}
function candidateCard(item) {
  const adopted = fitState.fit.adoption?.candidate_id === item.id;
  const status = item.outline_overlap
    ? badge("Outline overlap", "caution")
    : item.passes ? badge("Clears thresholds", "ready") : badge("Fails thresholds", "caution");
  return `<article class="card stack candidate ${adopted ? "adopted" : ""}">
    <div class="title-row"><h3>${e(item.label)}</h3>${adopted ? badge("Adopted base map", "ready") : status}</div>
    <div class="metrics">
      ${stat("Matched broken edge", `${n(item.broken_edge_matched_mm, 2)} / ${n(item.broken_edge_total_mm, 2)} mm`)}
      ${stat("Contacting solid edge", `${n(item.solid_contact_mm, 2)} mm`)}
      ${stat("Max colour-line offset", `${n(item.color_line_offset_max_mm, 2)} mm`)}
    </div>
    <p class="soft">Lateral colour-line offsets: ${item.color_line_offsets_mm.map(v => `${n(v, 2)} mm`).join(", ")}</p>
    <div class="actions">
      ${btn("Preview placement", "fit-candidate", "tiny secondary", "compare", `data-id="${e(item.id)}"`)}
    </div>
  </article>`;
}

function fitListPage() {
  const head = `<header class="page-head"><div><p class="eyebrow">Trial assembly</p><h1>Trial-fit base maps</h1>
    <p>Keep both fragments at their registered millimetre size, test two candidate placements and adopt a single base map for the handoff check.</p></div>
    <div class="actions">${btn("Load sample pair", "fit-sample", "light", "sample")}${btn("New trial fit", "fit-new", "primary", "plus")}</div></header>`;
  if (!fitState.fits.length) {
    return head + empty("No trial fit yet. Load the synthetic two-fragment pair to see a passing and a failing candidate.",
      btn("Load sample pair", "fit-sample", "light", "sample"));
  }
  const rows = fitState.fits.map(item => `<button class="tile-card" type="button" data-do="fit-open" data-id="${e(item.id)}">
    <div class="tile-cover">${item.base_map_digest ? img(`/api/fits/${item.id}/base-map.png`, item.title) : icon("frame")}</div>
    <div class="tile-info"><h3>${e(item.title)}</h3>
      <p>${e(item.fragment_a)} + ${e(item.fragment_b)}</p>
      <div class="tile-meta"><span>${item.adopted ? badge("Base map adopted", "ready") : badge("No candidate adopted", "caution")}</span><span>${date(item.updated_at)}</span></div>
    </div></button>`).join("");
  return `${head}<div class="collection">${rows}</div>`;
}

function thresholdSummary(fit) {
  const t = fit.thresholds;
  return table(["Threshold", "Value"], [
    ["Minimum matchable broken edge", `${n(t.minimum_matched_mm, 2)} mm`],
    ["Maximum colour-line offset at break", `${n(t.maximum_line_offset_mm, 2)} mm`],
    ["Empty seam width", `${n(t.seam_gap_mm, 2)} mm`],
  ]);
}

function fragmentSummary(side) {
  const member = fitState.fit.fragments[side];
  return `<div class="row"><div class="grow"><strong>${side} · ${e(member.tile_name)}</strong><br>
    <small>${e(member.mm_per_pixel.map(v => n(v, 4)).join(" × "))} mm/px · registered photo retained</small></div>
    ${btn("View marked photo", "fit-marked", "tiny secondary", "camera", `data-side="${side}"`)}</div>`;
}

function scanRows() {
  if (!fitState.scans.length) return "<p class='soft'>Draw a line across the fracture on the adopted base map.</p>";
  return `<div class="divider-list scroll-list">${fitState.scans.map(item => `<div class="row">
    <div class="grow"><strong>${e(item.title)}</strong><br><small>${item.metrics.supported_stations} readable · ${item.metrics.blank_stations.length} blank seam stations · ${date(item.created_at)}</small></div>
    ${btn("Open", "scan-open", "tiny secondary", null, `data-id="${e(item.id)}`)}</div>`).join("")}</div>`;
}

function scanDetail() {
  const scan = fitState.scan;
  if (!scan) return "";
  const m = scan.metrics;
  const series = [{values: m.rows.map(r => r.lightness), color: "#075d5a"}];
  const rows = m.rows.map(r => [
    r.station,
    r.side ?? "blank seam",
    r.samples,
    r.lightness == null ? "—" : n(r.lightness, 2),
    r.a == null ? "—" : n(r.a, 2),
    r.b == null ? "—" : n(r.b, 2),
  ]);
  return `<section class="card stack">
    <div class="title-row"><h3>${e(scan.title)}</h3>${badge(`${m.supported_stations} readings · ${m.blank_stations.length} blank`, "ready")}</div>
    <figure class="stage">${img(`/api/scans/${scan.id}/image/overlay`, "Fracture line scan overlay")}</figure>
    <div class="metrics">${stat("Line length", `${n(m.line_length, 2)} mm`)}${stat("Readable stations", m.supported_stations)}${stat("Blank seam stations", m.blank_stations.length)}</div>
    ${plot(series, "Lightness along the fracture line", {axis: `0–${n(m.line_length)} mm`})}
    <p class="notice">Blank stations sit in the empty seam and keep no colour even where the sampling band sees colour on both sides. Select a readable station to locate it on its original photo.</p>
    <details open><summary>Station readings and source side</summary>${table(["Station", "Source", "Samples", "L*", "a*", "b*"], rows)}</details>
    <div class="actions">${btn("Locate a station on its photo", "scan-locate", "secondary", "search")}</div>
  </section>`;
}

function fitDetailPage() {
  const fit = fitState.fit;
  const adopted = fit.adoption;
  return `<header class="page-head"><div><p class="eyebrow">Trial assembly</p><h1>${e(fit.title)}</h1>
    <p class="soft">${e(fit.note || "Two candidate placements evaluated against the same thresholds.")}</p></div>
    <div class="actions">${btn("Back to fits", "fit-back", "secondary", "back")}</div></header>
  <div class="grid-detail">
    <div class="stack">
      ${adopted ? `<section class="card stack"><h3>Adopted trial-fit base map</h3>
        <figure class="stage">${img(`/api/fits/${fit.id}/base-map.png?v=${fit.updated_at}`, "Adopted trial fit base map")}</figure>
        <p class="soft">Rule: ${e(adopted.rule.replaceAll("-", " "))}. Candidate <strong>${e(adopted.candidate_id)}</strong> wins on longest contacting solid edge, then closest colour-line registration.</p>
        <div class="actions">${btn("Draw fracture line scan", "scan-new", "primary", "measure")}</div></section>`
        : `<section class="card">${empty("No candidate cleared both thresholds, so no base map was adopted.")}</section>`}
      ${scanDetail()}
    </div>
    <div class="stack">
      <section class="card"><h3>Fragments (physical size preserved)</h3><div class="divider-list">${fragmentSummary("A")}${fragmentSummary("B")}</div></section>
      <section class="card"><h3>Conservator thresholds</h3>${thresholdSummary(fit)}</section>
      <section class="card stack"><h3>Candidate placements</h3><div class="stack">${fit.candidates.map(candidateCard).join("")}</div></section>
      <section class="card stack"><h3>Fracture line scans</h3>${scanRows()}</section>
    </div>
  </div>`;
}

export function renderFitPage() {
  q("#main").innerHTML = fitState.page === "fits" ? fitListPage() : fitDetailPage();
}

// ---------------------------------------------------------------------------
// Modals
// ---------------------------------------------------------------------------

function specimenOptions(selected) {
  return fitState.specimens.map(s => option(s.id, `${s.name} · ${s.width_mm ?? "?"} × ${s.height_mm ?? "?"} mm`, selected)).join("");
}

function pointInputs(name, value) {
  return `<div class="field"><span>${e(name)} (X, Y)</span><div class="point-group">
    <input type="number" min="0" max="1" step="0.001" value="${value?.[0] ?? 0.5}" data-point-x>
    <input type="number" min="0" max="1" step="0.001" value="${value?.[1] ?? 0.5}" data-point-y></div></div>`;
}

function newFitModal() {
  if (fitState.specimens.length < 2) {
    toast("Create two registered specimens with millimetre dimensions first, or load the sample pair.", true);
    return;
  }
  const body = `
    ${textField("Title", "title", "Trial fit · two fragments", 'required maxlength="110"')}
    <div class="form-columns">${selectField("Fragment A specimen", "tile_a", specimenOptions(fitState.specimens[0].id))}
      ${selectField("Fragment B specimen", "tile_b", specimenOptions(fitState.specimens[1].id))}</div>
    <p class="notice">Each fragment uses its registered base photo. Outline, solid/broken edges and colour lines use normalized photo coordinates (0–1); the demo payload below is pre-filled for the sample pair.</p>
    <div class="form-columns">
      ${numField("Min matched broken edge (mm)", "minimum_matched_mm", 20, 0.01, 100000, 0.1)}
      ${numField("Max colour-line offset (mm)", "maximum_line_offset_mm", 2.5, 0.001, 100000, 0.1)}
      ${numField("Empty seam width (mm)", "seam_gap_mm", 2, 0, 1000, 0.1)}
    </div>
    ${field("Candidate marks JSON (outline, edges, colour lines, anchors)", "payload_json",
      `<textarea id="field-payload_json" name="payload_json" style="min-height:230px"></textarea>`)}
    <p class="soft">Paste the two-fragment mark document. The sample pair button fills a working example.</p>`;
  openModal("New trial fit", body, "Evaluate candidates", async (data) => {
    const marks = JSON.parse(data.get("payload_json"));
    const specimens = Object.fromEntries(fitState.specimens.map(s => [s.id, s]));
    const tileA = specimens[data.get("tile_a")];
    const tileB = specimens[data.get("tile_b")];
    const payload = buildFitPayload(data, tileA, tileB, marks);
    const res = await api("/api/fits", {method: "POST", body: payload});
    closeModal();
    await openFit(res.fit.id);
    toast("Trial fit evaluated and one base map adopted.");
  }, true);
  q("#field-payload_json").value = JSON.stringify(sampleMarksDocument(), null, 2);
}

function buildFitPayload(data, tileA, tileB, marks) {
  const photoA = tileA.photos?.find?.(p => p.registration_state === "base")?.id ?? marks.photo_a;
  const photoB = tileB.photos?.find?.(p => p.registration_state === "base")?.id ?? marks.photo_b;
  return {
    title: data.get("title"),
    note: marks.note || "",
    fragments: {
      A: {tile_id: tileA.id, photo_id: photoA, marks: marks.A},
      B: {tile_id: tileB.id, photo_id: photoB, marks: marks.B},
    },
    candidates: marks.candidates,
    thresholds: {
      minimum_matched_mm: Number(data.get("minimum_matched_mm")),
      maximum_line_offset_mm: Number(data.get("maximum_line_offset_mm")),
      seam_gap_mm: Number(data.get("seam_gap_mm")),
    },
  };
}

function sampleMarksDocument() {
  // Mirrors frescobench/trial_pair.py so the prefilled UI sample evaluates the
  // same passing and failing candidates as the synthetic training pair.
  const lineY = 0.5;
  const outlineA = [[0.03, 0.07], [0.95, 0.07], [0.95, 0.93], [0.03, 0.93]];
  const outlineB = [[0.97, 0.07], [0.05, 0.07], [0.05, 0.93], [0.97, 0.93]];
  const A = {
    outline: outlineA,
    edges: {
      solid: [[[0.03, 0.06], [0.03, 0.94]], [[0.04, 0.04], [0.96, 0.04]], [[0.96, 0.96], [0.04, 0.96]]],
      broken: [[[0.977, 0.06], [0.977, 0.94]]],
    },
    color_lines: [[[0.12, lineY], [0.96, lineY]]],
  };
  const B = {
    outline: outlineB,
    edges: {
      solid: [[[0.97, 0.06], [0.97, 0.94]], [[0.04, 0.04], [0.96, 0.04]], [[0.96, 0.96], [0.04, 0.96]]],
      broken: [[[0.023, 0.94], [0.023, 0.06]]],
    },
    color_lines: [[[0.88, lineY], [0.04, lineY]]],
  };
  // Equal inset from each broken edge, matching the backend training pair.
  const anchorA = 0.952;
  const anchorB = 0.048;
  return {
    note: "Candidate 1 closes the fracture; candidate 2 shifts fragment B down by 3.6 mm.",
    A, B,
    candidates: [
      {id: "close", label: "Candidate 1 · fracture closed",
        anchors_a: [[anchorA, 0.12], [anchorA, 0.86]],
        anchors_b: [[anchorB, 0.12], [anchorB, 0.86]]},
      {id: "shifted", label: "Candidate 2 · shifted 3.6 mm down",
        anchors_a: [[anchorA, 0.12], [anchorA, 0.86]],
        anchors_b: [[anchorB, 0.24], [anchorB, 0.98]]},
    ],
  };
}

function scanModal() {
  const body = `${pointInputs("Line start (base map fraction)", [0.18, 0.47])}
    ${pointInputs("Line end (base map fraction)", [0.82, 0.47])}
    <div class="form-columns">${numField("Stations", "samples", 120, 20, 360, 1)}
      ${numField("Band width px", "band_pixels", 5, 0.5, 32, 0.5)}</div>
    <p class="notice">Draw across the fracture on the adopted base map. Stations whose center lands in the empty seam stay blank and are not assigned a colour.</p>`;
  openModal("Fracture line scan", body, "Save line scan", async (data) => {
    const xs = qa("[data-point-x]", q("#dialog-form")).map(el => Number(el.value));
    const ys = qa("[data-point-y]", q("#dialog-form")).map(el => Number(el.value));
    const scan = await api(`/api/fits/${fitState.fit.id}/scans`, {method: "POST", body: {
      title: "Fracture line scan",
      start: [xs[0], ys[0]], end: [xs[1], ys[1]],
      samples: Number(data.get("samples")), band_pixels: Number(data.get("band_pixels")),
    }});
    closeModal();
    fitState.scans = (await api(`/api/fits/${fitState.fit.id}/scans`)).items;
    fitState.scan = scan;
    renderFitPage();
    toast("Fracture line scan saved with source-side attribution.");
  }, true);
}

function locateStationModal() {
  const scan = fitState.scan;
  const readable = scan.metrics.rows.filter(r => r.side);
  const body = selectField("Station", "station", readable.map(r =>
    option(r.station, `${r.station} · fragment ${r.side} · L* ${r.lightness == null ? "—" : n(r.lightness)}`)).join(""));
  openModal("Locate a reading on its original photo", body, "Open source", async (data) => {
    const station = Number(data.get("station"));
    const info = await api(`/api/fits/${fitState.fit.id}/scans/${scan.id}/stations/${station}`);
    closeModal();
    showStationCard(info);
  });
}

function showStationCard(info) {
  const member = fitState.fit.fragments[info.side];
  const url = `/api/specimens/${member.tile_id}/viewer/${info.photo_id}?layer=raw`;
  openModal(`Station ${info.station} · fragment ${info.side}`, `
    <figure class="stage">${img(url, `Original registered photo for fragment ${info.side}`)}</figure>
    <div class="metrics">${stat("Specimen", info.tile_name)}${stat("Local X / Y", `${n(info.local_position[0], 3)}, ${n(info.local_position[1], 3)}`)}${stat("Lightness L*", n(info.lightness, 2))}</div>
    <p class="notice">This reading is taken from <strong>${e(info.photo_name)}</strong>. The local position marks the same point on that original registered photo, not on a composite preview.</p>`,
    "", null, true);
}

async function candidatePreviewModal(candidateId) {
  const token = ++fitState.token;
  const res = await request(`/api/fits/${fitState.fit.id}/candidate-preview`, {method: "POST", body: {candidate_id: candidateId}});
  const blob = await res.blob();
  const uri = URL.createObjectURL(blob);
  if (token !== fitState.token) { URL.revokeObjectURL(uri); return; }
  openModal("Candidate placement preview", `<figure class="stage">${img(uri, "Candidate trial fit placement")}</figure>
    <p class="soft">Both fragments stay at their registered physical size; the empty seam shows between them.</p>`, "", null, true);
}

async function markedPhotoModal(side) {
  const member = fitState.fit.fragments[side];
  const token = ++fitState.token;
  const res = await request("/api/fits/marked-photo", {method: "POST", body: {tile_id: member.tile_id, photo_id: member.photo_id, marks: member.marks}});
  const blob = await res.blob();
  const uri = URL.createObjectURL(blob);
  if (token !== fitState.token) { URL.revokeObjectURL(uri); return; }
  openModal(`Fragment ${side} · marked registered photo`, `<figure class="stage">${img(uri, "Marked fragment photo")}</figure>
    <p class="soft">Outline, solid and broken edges and fracture-facing colour line ends on the millimetre-registered photo.</p>`, "", null, true);
}

export async function fitAction(name, target) {
  const {id, side} = target.dataset;
  switch (name) {
    case "fit-sample": {
      const res = await api("/api/training/trial-pair", {method: "POST", body: {}});
      await openFit(res.fit_id);
      toast("Synthetic fragment pair and evaluated trial fit loaded.");
      break;
    }
    case "fit-new": newFitModal(); break;
    case "fit-open": await openFit(id); break;
    case "fit-back": await openFitList(); break;
    case "fit-candidate": await candidatePreviewModal(id); break;
    case "fit-marked": await markedPhotoModal(side); break;
    case "scan-new": scanModal(); break;
    case "scan-open":
      fitState.scan = await api(`/api/scans/${id}`);
      renderFitPage();
      break;
    case "scan-locate": locateStationModal(); break;
    default:
      break;
  }
}
