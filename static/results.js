import {badge, btn, e, img, link, n, percent, plot, stat, table, date, dots} from "./widgets.js";

const colors = ["#58836b","#b26a4a","#818d40","#786f99"];
const layers = {overlay:"Measurement map",source:"Source plane",paired:"Paired plane",reference:"Reference image"};

export function metricSet(record) {
  const m = record.metrics;
  const rows = {
    color: () => [["Median L*",n(m.median_lab[0])],["Within-plane p90 ΔE",n(m.delta_e_from_tile_median["90"])],["Pixels measured",n(m.pixels,0)]],
    pores: () => [["Closed candidates",m.closed_count],["Truncated candidates",m.truncated_count],["Closed dark area",percent(m.closed_dark_percent)]],
    dots: () => [["Resolved grains",m.dot_count],["Median area (px²)",n(m.median_area_px,2)],["Resolved grain area",percent(m.ink_area_percent)]],
    weave: () => [["Family A pitch",m.axis_a_pitch==null?"—":`${n(m.axis_a_pitch,3)} ${m.unit}`],["Family B pitch",m.axis_b_pitch==null?"—":`${n(m.axis_b_pitch,3)} ${m.unit}`],["Crossing",m.crossing_angle==null?"—":`${n(m.crossing_angle,1)}°`]],
    coverage: () => [["Intact pigment",percent(m.percent.glazed)],["Bare plaster",percent(m.percent.bare)],["Mixed",percent(m.percent.mixed)],["Unmeasured",percent(m.percent.unmeasured)]],
    salt: () => [["Candidate deposits",m.component_count],["Touching support boundary",m.boundary_count],["Share of observed area",percent(m.candidate_percent)]],
    defects: () => {
      const reviewed = Object.values(record.decisions);
      return [["Observed",m.total_observations],["Confirmed",reviewed.filter(item=>item.decision==="confirm").length],["Dismissed",reviewed.filter(item=>item.decision==="dismiss").length]];
    },
    glare: () => [["Strong glare",percent(m.strong_glare_percent)],["Counted hotspots",m.hotspot_count],["Paired support",percent(m.support.overlap_percent)]],
    direction: () => [["Left-bright detail",percent(m.left_strong_percent)],["Right-bright detail",percent(m.right_strong_percent)],["Paired support",percent(m.support.overlap_percent)]],
    texture: () => [["Medium local roughness",n(m.grid_roughness["50"],5)],["Ridge axis",m.texture_orientation ? n(m.texture_orientation.ridge_angle_clockwise)+"°" : "—"],["Coherence",n(m.texture_orientation?.coherence,3)]],
    craze: () => [["Trace length",`${n(m.network_length)} ${m.length_unit}`],["Junctions",n(m.junctions,0)],["Closed cells",n(m.closed_cell_count,0)]],
    rim: () => [["Rim median ΔE",n(m.all_edges.median_from_center_delta_e)],["Rim pixels",n(m.all_edges.pixels,0)],["Interior L*",n(m.interior.median_lab[0])]],
    profile: () => [["Distance",`${n(m.line_length)} ${m.distance_unit}`],["Valid stations",`${m.supported_stations}/${m.requested_stations}`],["Median L*",n(m.features.median_lightness)]],
    reference: () => [["Specimen to reference ΔE",n(m.delta_e76)],["Lightness difference",n(m.delta_lab[0])],["Dispersion difference",n(m.dispersion_delta)]],
    marking: () => [["Measured stations",percent(m.observed_percent)],["Rule among measured",percent(m.midrib_percent_of_observed)],["Runs",m.runs.length]],
    joint: () => [["Measured stations",m.measured_stations],["Unmeasured stations",m.unmeasured_stations],["Median absolute contrast",n(m.median_absolute_step,4)]],
    scribe: () => [["Median exposure width",m.median_width==null?"—":`${n(m.median_width,3)} ${m.length_unit}`],["Maximum exposure width",m.maximum_width==null?"—":`${n(m.maximum_width,3)} ${m.length_unit}`],["Both sides measured",`${m.observed_stations}/${m.requested_stations}`]],
    aggregate: () => [["Bright clusters",m.candidate_clusters],["Supported pixels",m.observed_pixels],["Bright fraction",percent(m.bright_fraction_of_support)]],
  }[record.kind]();
  return `<div class="metrics">${rows.map(([name,value])=>stat(name,value)).join("")}</div>`;
}

function legend(record) {
  if (!record.legend) return "";
  const scale=record.legend;
  if (scale.mode==="categorical") {
    return `<div class="legend"><span>Image classification</span><div class="swatches">${scale.labels.map(label=>dots(label.name,label.color)).join("")}</div></div>`;
  }
  return `<div class="legend"><span>${e(scale.label)}</span><span>${n(scale.low,4)} to ${n(scale.high,4)}</span></div><div class="gradient ${e(scale.mode)}"></div>`;
}

function colorDetail(m) {
  return `<h3>Color palette</h3><div class="swatches">${m.palette.map(item=>dots(percent(item.coverage_percent), item.swatch)).join(" ")}</div>
    <details><summary>Spatial cells</summary>${table(["Column, row","Pixels","Median Lab","From median ΔE"], m.cells.map(item=>[item.cell.join(", "),item.pixels,item.lab.map(val=>n(val)).join(" / "),n(item.delta_from_center)]))}</details>`;
}

function coverageDetail(m) {
  const series = ["glazed","mixed","bare"].map((topic,index)=>({values:m.depth_bands.map(item=>item.percent[topic]),color:colors[index]}));
  return `<h3>Coverage across the sample</h3>${plot(series,"Coverage from top to bottom",{min:0,max:100})}
  ${table(["Depth band","Intact pigment","Mixed","Bare plaster","Unmeasured"],m.depth_bands.map(item=>[`${n(item.relative_top*100,0)}–${n(item.relative_bottom*100,0)}%`,...["glazed","mixed","bare","unmeasured"].map(key=>percent(item.percent[key]))]))}`;
}

function markingDetail(m) {
  const rows=m.runs.map(run=>[run.condition,n(run.first_station,1),n(run.last_station,1),run.stations,n(run.median_cross_width,1)]);
  return `<h3>Edge mark visibility runs</h3><p class="plot-note">${e(m.note)}</p>
    ${table(["Condition",`From (${m.unit})`,`To (${m.unit})`,"Stations",`Median width (${m.unit})`],rows)}`;
}

function jointDetail(m) {
  const rows=m.stations.map(item=>[n(item.position,1),n(item.left,4),n(item.right,4),n(item.right_minus_left,4),item.status]);
  return `<h3>Contrast along the mounting seam</h3><p class="plot-note">${e(m.note)}</p>
    ${plot([{values:m.stations.map(item=>item.right_minus_left)}],"Right minus left lightness")}
    <details><summary>Station readings</summary>${table([`Distance (${m.unit})`,"Left","Right","Difference","Condition"],rows)}</details>`;
}

function aggregateDetail(m) {
  const rows=m.candidates.map(item=>[item.rank,item.registered_pixels,n(item.contrast,4),n(item.bounding_width,1),n(item.bounding_height,1)]);
  return `<h3>Pale fleck candidates</h3><p class="plot-note">${e(m.note)}</p>
    ${table(["Rank","Pixels","Peak contrast",`Width (${m.unit})`,`Height (${m.unit})`],rows)}`;
}

function scribeDetail(m) {
  const unit=m.length_unit;
  const rows=m.stations.map(station=>[n(station.position,2),n(station.left.width,2),station.left.state,n(station.right.width,2),station.right.state,n(station.total_width,2)]);
  const series=[
    {values:m.stations.map(station=>station.left.width),color:"#067c72"},
    {values:m.stations.map(station=>station.right.width),color:"#4168ab"},
  ];
  return `<h3>Exposure from the marked loss axis</h3><p>${e(m.note)}</p>
    <div class="swatches">${dots("Left of scribe",series[0].color)} ${dots("Right of scribe",series[1].color)}</div>
    ${plot(series,"Exposure width along scribe",{min:0})}
    <details><summary>Station readings</summary>${table([`Distance (${unit})`,`Left (${unit})`,"Left evidence",`Right (${unit})`,"Right evidence",`Total (${unit})`],rows)}</details>`;
}

function defectsDetail(record, payload) {
  const markers=record.metrics.observations;
  const continuation=payload?.continuation;
  const earlierRows=continuation?.archive.earlier || [];
  const earlierByLater=new Map(earlierRows.filter(row=>row.later_key).map(row=>[row.later_key,row]));
  if (!markers.length) return "<p>No spots exceeded this run's current thresholds.</p>";
  return `<h3>Review local observations</h3>
    <div class="divider-list scroll-list">${markers.map(marker=>{
      const decision=record.decisions[marker.key]?.decision || "review";
      const carried=earlierByLater.get(marker.key);
      return `<div class="row"><div class="grow"><strong>${e(marker.key)} · ${e(marker.type)}</strong><br><small>${n(marker.area_pixels,0)} px · ${e(marker.polarity)} · ${e(decision)}${carried?` · continues earlier ${e(carried.key)} (${e(carried.previous_decision)}${carried.previous_notes?` · ${e(carried.previous_notes)}`:""})`:""}</small></div>
        ${btn("Confirm","defect-choice","tiny secondary",null,`data-marker="${e(marker.key)}" data-value="confirm"`)}
        ${btn("Dismiss","defect-choice","tiny ghost",null,`data-marker="${e(marker.key)}" data-value="dismiss"`)}
        ${btn("Note","defect-note","tiny ghost",null,`data-marker="${e(marker.key)}"`)}</div>`;
    }).join("")}</div>`;
}

function limitText(limits) {
  return `center offset ≤ ${n(limits.center_offset_px)} px · footprint overlap ≥ ${n(limits.footprint_overlap)}`;
}

function successorPanel(payload) {
  const list=payload?.successors || [];
  if (!list.length) return "";
  return `<h3>Feeds later runs</h3>
    ${list.map(next=>`<p class="soft">This run continues into the ${e(next.photo_role)} run of ${e(date(next.created_at))}: ${next.continued} of ${next.earlier_total} earlier spots continued there.</p>
    <div class="actions">${btn("Open the later run","open-successor","secondary","arrow",`data-id="${e(next.id)}"`)}</div>`).join("")}`;
}

function continuationPanel(record, payload, currentRevision) {
  if (!payload) return `<h3>Continuation from an earlier run</h3><p class="plot-note">Loading saved continuation…</p>`;
  const back=payload.back_to ? `<div class="actions">${btn("Back to the later run","close-earlier","tiny secondary","back")}</div>` : "";
  const continuation=payload.continuation;
  if (!continuation) {
    const options=payload.earlier_options || [];
    if (!options.length) return `<h3>Continuation from an earlier run</h3>${back}
      <p class="soft">Run Surface observations on a second registered photograph of this specimen, review its spots, then return here.</p>`;
    return `<h3>Continuation from an earlier run</h3>${back}
      <p class="soft">Carry reviewed spots from an earlier run on another photograph of this specimen. A pair qualifies only where both runs photograph it and it meets the limits you enter.</p>
      <div class="actions">${btn("Continue from earlier run","continuation-start","primary","compare")}</div>`;
  }
  const summary=continuation.summary;
  const earlier=payload.earlier_measurement;
  const warning=payload.warning?`<p class="notice warn">${e(payload.warning)}</p>`:"";
  const rows=continuation.archive.earlier;
  const optionsByEarlier=new Map();
  for (const pair of payload.candidate_pairs || []) {
    if (!optionsByEarlier.has(pair.earlier)) optionsByEarlier.set(pair.earlier, []);
    optionsByEarlier.get(pair.earlier).push(pair);
  }
  const changes=(continuation.history || []).filter(entry=>entry.action!=="resuggested");
  const portrait=earlier?`<figure class="stage">${img(earlier.photo.overlay_url,`Earlier run ${e(earlier.photo.role)}`)}</figure>
    <p class="plot-note">Opened from the earlier run: ${e(earlier.photo.role)} · ${e(earlier.photo.name)} · ${e(date(earlier.created_at))}.</p>`:"";
  return `<h3>Continuation from an earlier run</h3>${warning}${back}
    <div class="metrics">${stat("Continued",`${summary.continued} of ${summary.earlier_total}`)}${stat("Not photographed",summary.earlier_not_photographed)}${stat("No qualifying spot",summary.earlier_no_qualifying_continuation)}${stat("Confirmed kept",summary.confirmed_continued)}${stat("Confirmed archived",summary.confirmed_archived)}</div>
    <p class="plot-note">Earlier run: ${e(continuation.earlier_photo.role)} · ${e(continuation.earlier_photo.name)} · ${e(date(continuation.earlier_created_at))}. Limits: ${e(limitText(continuation.limits))}.</p>
    <p class="plot-note">${e(continuation.rules.ranking)} ${e(continuation.rules.capture)}</p>
    <div class="actions">${btn("Open earlier run","open-earlier","tiny secondary","book")}
      ${link("Earlier overlay","/api/results/"+e(continuation.earlier_id)+"/image/overlay","tiny ghost","download")}
      ${btn("Rebuild automatic relation","continuation-start","tiny ghost","compare")}</div>
    <div class="divider-list scroll-list">${rows.map(row=>{
      const choices=optionsByEarlier.get(row.key) || [];
      const canChange=row.outcome!=="not_photographed" && (choices.length || row.outcome==="continued");
      const aside=row.set_aside?" · kept out by hand":"";
      const detail=row.outcome==="continued"
        ? `continues ${row.later_key} · earlier ${e(row.previous_decision)} · offset ${n(row.center_offset_px)} px · footprint ${n(row.footprint_overlap,3)}`
        : `${e(row.outcome_label)}${aside} · earlier ${e(row.previous_decision)}${row.previous_notes?` · ${e(row.previous_notes)}`:""}`;
      return `<div class="row"><div class="grow"><strong>${e(row.key)}</strong><br><small>${e(detail)}</small></div>
        ${canChange?btn(row.outcome==="continued"?"Change":"Attach","continuation-link","tiny ghost",null,`data-earlier="${e(row.key)}"`):""}</div>`;
    }).join("")}</div>
    ${portrait}
    ${changes.length?`<details><summary>Change history (${changes.length})</summary>${table(["When","Action","Earlier","From","To","Reason"],changes.map(entry=>[date(entry.changed_at),entry.action,entry.earlier||"—",entry.from||"—",entry.to||"—",entry.reason||"—"]))}</details>`:""}`;
}


function textureDetail(m) {
  return `<h3>Texture repeats by direction</h3>
    ${plot(m.lag_curves.map((dir,index)=>({values:dir.points.map(point=>point.correlation),color:colors[index]})),"Spacing along each direction",{min:-1,max:1})}
    <div class="swatches">${m.lag_curves.map((dir,index)=>dots(dir.direction,colors[index])).join(" ")}</div>
    ${table(["Direction","Peak spacing","Correlation"],m.lag_curves.flatMap(dir=>dir.local_highs.map(point=>[dir.direction,point.spacing_mm ? `${n(point.spacing_mm)} mm` : `${n(point.spacing_px)} px`,n(point.correlation,3)])))}
    <p class="plot-note">${e(m.note)}</p>`;
}

function profileDetail(m) {
  return `<h3>Lightness along the marked line</h3>${plot([{values:m.rows.map(item=>item.lightness)}], "Lightness along the station band",{axis:`0–${n(m.line_length)} ${m.distance_unit}`})}
    <details><summary>Sample station values</summary>${table(["Distance","Valid samples","L*","a*","b*"],m.rows.map(item=>[`${n(item.position,3)} ${item.position_unit}`,item.samples,n(item.lightness),n(item.a),n(item.b)]))}</details>`;
}

function glareDetail(m) {
  return `<h3>Glare hotspots</h3><p class="plot-note">${e(m.note)} Balancing gain ${n(m.gain,4)}.</p>
    ${table(["Hotspot","Pixels","Area mm²"],m.hotspots.map((item,index)=>[index+1,item.pixels,n(item.area_mm2,3)]))}`;
}

function directionDetail(m) {
  return `<h3>Raking contrast by cell</h3><p class="plot-note">${e(m.note)}</p>
    ${table(["Column, row","Signed median","Strong contrast"],m.cells.map(item=>[item.cell.join(", "),n(item.median_signed,4),percent(item.strong_percent)]))}`;
}

function crazeDetail(m) {
  return `<h3>Closed lines</h3><p class="plot-note">${e(m.note)}</p>
  ${table(["Cell","Pixels","Area mm²"],m.largest_cells.map((item,index)=>[index+1,item.pixels,n(item.mm2,3)]))}`;
}

function rimDetail(m) {
  return `<h3>Rim color by side</h3><p class="plot-note">${e(m.note)}</p>
  ${table(["Side","Pixels","Median from interior ΔE","Excluded"],m.sides.map(item=>[item.side,item.stats?.pixels,n(item.stats?.median_from_center_delta_e),percent(item.excluded_percent)]))}`;
}

function referenceDetail(m) {
  return `<h3>${e(m.reference_title)}</h3>
    <div class="metrics">${stat("Specimen",m.specimen.swatch)}${stat("Approved reference",m.reference.swatch)}</div>
    <p class="notice">${e(m.capture_note)}</p>`;
}

function details(record, payload) {
  const render = {
    weave:m=>`<h3>Visible texture families</h3><p class="plot-note">${e(m.note)}</p>${table(["Family","Direction (°)",`Pitch (${m.unit})`,"Peak / background"],m.rows.map(item=>[item.family,n(item.angle_degrees,2),n(item.pitch,3),n(item.peak_to_background,2)]))}`,
    pores:m=>`<h3>Opening record</h3><p class="plot-note">${e(m.note)} ${m.features_omitted?`Table retains the largest ${m.rows.length} openings.`:""}</p>
      ${table(["Opening","State","Area (px²)","Area (mm²)","Position"],m.rows.map(r=>[r.id,r.state,r.pixels,n(r.square_mm,4),r.center_normalized.map(value=>n(value,3)).join(", ")]))}`,
    salt:m=>`<h3>Mapped deposits</h3><p class="plot-note">${e(m.note)} ${m.rows_omitted?`Table retains the largest ${m.rows.length} deposits.`:""}</p>
      <div class="swatches">${dots(m.control.label,m.control.swatch)}</div>
      ${table(["Deposit","Support","Area (px²)","Area (mm²)"],m.rows.map(r=>[r.id,r.state,r.pixels,n(r.square_mm,4)]))}`,
    dots:m=>`<h3>Isolated dark grains</h3><p class="plot-note">${e(m.note)}</p>${table(["Grain","Area (px²)","Area (mm²)","Major/minor","Angle (°)"],m.rows.map(r=>[
      r.dot, n(r.area_px,0), n(r.area_mm2,5), n(r.axis_ratio,2), n(r.major_angle_deg,1)
    ]))}`,
    color:colorDetail,coverage:coverageDetail,defects:record=>defectsDetail(record,payload),glare:glareDetail,
    direction:directionDetail,texture:textureDetail,profile:profileDetail,reference:referenceDetail,craze:crazeDetail,rim:rimDetail,
    marking:markingDetail,joint:jointDetail,aggregate:aggregateDetail,scribe:scribeDetail,
  }[record.kind];
  return render(record.kind==="defects" ? record : record.metrics);
}

export function resultCard(record, layer, currentRevision, payload) {
  const path=`/api/results/${record.id}`;
  const available = ["overlay","source",...(record.pictures.paired ? ["paired"] : []),...(record.metrics.reference_image ? ["reference"] : [])];
  const old = record.source_version !== currentRevision;
  const continuation = record.kind==="defects" ? continuationPanel(record, payload)+successorPanel(payload) : "";
  return `<section class="card stack">
    <div class="title-row"><div><h3>${e(record.label)}</h3><small>Specimen revision ${record.source_version} · ${e(date(record.created_at))}</small></div>${old ? badge("Earlier specimen revision","caution") : badge("Saved measurement","ready")}</div>
    <div class="actions">${available.map(key=>btn(layers[key],"result-layer",layer===key?"tiny primary":"tiny secondary",null,`data-value="${key}"`)).join("")}</div>
    <figure class="stage">${img(path+"/image/"+layer+"?v="+record.version,`${record.label} · ${layers[layer]}`)}</figure>
    ${layer==="overlay"?legend(record):""}
    ${metricSet(record)}
    <section>${details(record,payload)}</section>
    ${continuation}
    <details><summary>Measurement source</summary>${table(["Property","Value"],[
      ["Image",record.provenance.photo.display_name],["Image role",record.provenance.photo.role],["Region",record.provenance.region?.name || "Entire plane"],
      ["Photo digest",record.provenance.photo.original_digest],
      ["Supported area",percent(record.support.available_percent)],
      ["Clipped pixels",record.support.clipped_pixels],
      ["Neutral correction",record.provenance.neutral_card?.before_swatch || "Uncorrected"],
    ])}</details>
    <div class="actions">${link("Metrics CSV",path+"/metrics.csv","secondary","download")}${record.continuation?link("Continuation CSV",path+"/continuation.csv","secondary","download"):""}${btn("Pin evidence","pin", "primary","archive")}</div>
  </section>`;
}

export function specimenResultRows(items) {
  if (!items.length) return "<p class='soft'>Run a measurement to start this specimen’s history.</p>";
  return `<div class="divider-list scroll-list">${items.map(item=>`<div class="row"><div class="grow"><strong>${e(item.label)}</strong><br><small>Revision ${item.source_version} · ${e(date(item.created_at))}${item.continuation?` · continues ${item.continuation.continued} of ${item.continuation.earlier_total} ${e(item.continuation.earlier_photo_role)} spots`:""}</small></div>${btn("Open","result-open","tiny secondary",null,`data-id="${item.id}"`)}</div>`).join("")}</div>`;
}
