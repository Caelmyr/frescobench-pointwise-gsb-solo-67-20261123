# FrescoBench

FrescoBench is a local workbench for conservators inspecting photographed wall painting fragments. It retains original exposures, registers views on a measured plane and saves the support behind each reading.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19989`. Choose another local port with `FRESCOBENCH_PORT`. A named volume stores captures and measurements. With Python 3.11 and the pinned dependencies installed, `./start.sh` starts the same workbench.

## Inspect a fragment

1. Load the four synthetic sample fragments or add photos in **Captures**. Mark the four survey corners, enter the plane width and height in millimeters, and register each alternate exposure against the corresponding corners. In **Sampling regions**, distinguish inspected areas, material references and exclusions.
2. In **Measure**, use **Salt bloom candidates** on the painted face of *Ultramarine · W12*. It uses an intact pigment control to find connected, light, low-chroma deposits and marks candidates reaching a support boundary. **Loss margin width** measures bare plaster on each side of a marked loss strip. **Paint coverage** separates bare, painted and ambiguous areas. **Plaster pore candidates** distinguishes closed dark footprints from truncated ones. Each measurement retains a map and downloadable readings.
3. Use **Surface observations** to review individual marks, **Polarized glare** and **Raking light comparison** for photograph pairs, **Pigment texture spacing** and **Fine-line network** for close inspection, and **Geometry** for physical traces and areas. **Fragment studies** compares retained measurements. **Evidence** builds a ZIP of selected measurements and optional originals. **Received evidence** verifies the declared data in an opened evidence ZIP.

   In **Geometry**, mark foot-pad candidates on each side of the loss, the supportable bare plaster, the no-pressure zone and the painted area that must stay visible in **Sampling regions**, then **Adopt a pad pair**. Every candidate pad counts only the largest single connected, evidently visible bearing patch; separated patches are never added together. Pads must not overlap or enter a no-pressure zone, each side must reach its minimum bearing area, and the adopted pair must leave the visible-paint minimum. When one pair bears more and another hides less paint, the fixed field rule is *painting first*: among pairs meeting all minimums, adopt the pair covering the least visible paint; equal coverage goes to the larger combined acknowledged bearing area, then the earlier-marked candidate. The adopted pair and its regions are frozen and drawn for the installer; re-adopt or release the pair to edit them. **Trace actual footing** then reads a later line against the frozen pair only, separating continuous runs on each acknowledged bearing piece from interruptions across the pad gap or under parts of a pad that were never recognized as bearing.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A specimen holds at most 10 captures and 24 regions. Registration and distances are planar. The longest side of the registered sampling plane is limited to 1,800 pixels. Deposit results guide closer inspection and do not establish chemical composition. The sample fragments and exposures are synthetic.

FrescoBench builds on the local CoreSection capture, registration, analysis and evidence libraries. Its wall-fragment imagery and deposit analysis ship with the application.
