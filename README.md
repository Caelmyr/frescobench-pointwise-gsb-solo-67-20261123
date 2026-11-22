# FrescoBench

FrescoBench is a local workbench for conservators inspecting photographed wall painting fragments. It retains original exposures, registers views on a measured plane and saves the support behind each reading.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19989`. Choose another local port with `FRESCOBENCH_PORT`. A named volume stores captures and measurements. With Python 3.11 and the pinned dependencies installed, `./start.sh` starts the same workbench.

## Inspect a fragment

1. Load the four synthetic sample fragments or add photos in **Captures**. Mark the four survey corners, enter the plane width and height in millimeters, and register each alternate exposure against the corresponding corners. In **Sampling regions**, distinguish inspected areas, material references and exclusions.
2. In **Measure**, use **Salt bloom candidates** on the painted face of *Ultramarine · W12*. It uses an intact pigment control to find connected, light, low-chroma deposits and marks candidates reaching a support boundary. **Loss margin width** measures bare plaster on each side of a marked loss strip. **Paint coverage** separates bare, painted and ambiguous areas. **Plaster pore candidates** distinguishes closed dark footprints from truncated ones. Each measurement retains a map and downloadable readings.
3. Use **Surface observations** to review individual marks, **Polarized glare** and **Raking light comparison** for photograph pairs, **Pigment texture spacing** and **Fine-line network** for close inspection, and **Geometry** for physical traces and areas. **Fragment studies** compares retained measurements. **Evidence** builds a ZIP of selected measurements and optional originals. **Received evidence** verifies the declared data in an opened evidence ZIP.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A specimen holds at most 10 captures and 24 regions. Registration and distances are planar. The longest side of the registered sampling plane is limited to 1,800 pixels. Deposit results guide closer inspection and do not establish chemical composition. The sample fragments and exposures are synthetic.

## Trial-fit two fragments

In **Trial fits**, two separately registered fragments (each with millimetre dimensions) keep their physical size: the second fragment moves by rotation and translation only, never by scaling. On each fragment's registered photo mark the outline, the solid and broken edges, and the visible colour lines where they reach the break, then give two candidate placements through matching anchor pairs and set the two gates you decide: a minimum matchable broken-edge length (mm) and a maximum colour-line offset at the fracture (mm), plus the empty seam width. A candidate whose outlines overlap, whose matchable broken edge is too short, or whose colour line misses by more than the offset limit fails.

Among the candidates that clear both gates, the one with the longest physically contacting solid edge is adopted; if two contact the same length of solid edge, the closer colour-line registration wins, and that tie-break is recorded. Only one candidate becomes the single trial-fit base map handed over for the "same painted line" check; if no candidate clears the gates, no base map is stored.

A fracture **Line scan** is drawn on the adopted base map only. Each station is owned by the base-map pixel it lands on: stations in the empty seam stay explicitly blank with no colour even where the transverse band still sees colour on both sides, and every readable station is attributed to the fragment side it sits on, with a point that maps back onto that fragment's original registered photo. The adopted placement, the two source photos and every scan persist, so reopening the workspace shows exactly which fragment each reading came from rather than a composite preview.

FrescoBench builds on the local CoreSection capture, registration, analysis and evidence libraries. Its wall-fragment imagery and deposit analysis ship with the application.
