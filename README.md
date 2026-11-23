# FrescoBench

FrescoBench is a local workbench for conservators inspecting photographed wall painting fragments. It retains original exposures, registers views on a measured plane and saves the support behind each reading.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19989`. Choose another local port with `FRESCOBENCH_PORT`. A named volume stores captures and measurements. With Python 3.11 and the pinned dependencies installed, `./start.sh` starts the same workbench.

## Inspect a fragment

1. Load the four synthetic sample fragments or add photos in **Captures**. Mark the four survey corners, enter the plane width and height in millimeters, and register each alternate exposure against the corresponding corners. In **Sampling regions**, distinguish inspected areas, material references and exclusions.
2. In **Measure**, use **Salt bloom candidates** on the painted face of *Ultramarine · W12*. It uses an intact pigment control to find connected, light, low-chroma deposits and marks candidates reaching a support boundary. **Loss margin width** measures bare plaster on each side of a marked loss strip. **Paint coverage** separates bare, painted and ambiguous areas. **Plaster pore candidates** distinguishes closed dark footprints from truncated ones. Each measurement retains a map and downloadable readings.
3. Use **Surface observations** to review individual marks, **Polarized glare** and **Raking light comparison** for photograph pairs, **Pigment texture spacing** and **Fine-line network** for close inspection, and **Geometry** for physical traces and areas. **Fragment studies** compares retained measurements. **Evidence** builds a ZIP of selected measurements and optional originals. **Received evidence** verifies the declared data in an opened evidence ZIP.
4. For a before/after cleaning campaign, register one **Before cleaning** and one **After cleaning** photo of each equally sized fragment against the same millimetre plane (aligned on the fixed engraved lines), mark one bare-plaster control, one intact-pigment control and the inspection area. In **Fragment studies → Build cleaning study**, enter the minimum fragment-pair count and minimum common visible area. Every retained pair is measured only on positions every selected photo actually sees: a position outside a frame or clipped (over-exposed) in any selected photo is excluded for the whole study and is never back-filled from the other photos. The selection first maximizes the number of retained pairs, then maximizes the strictly common visible area (ties resolve by specimen ID), and the resulting pairs, region and photo set are frozen. Later, a **Cleaning study re-check** on a specimen measures bare plaster in a small box only where it intersects that frozen common region at positions the study observed; newly added photos and independent measurements never alter an adopted study.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A specimen holds at most 10 captures and 24 regions. Registration and distances are planar. The longest side of the registered sampling plane is limited to 1,800 pixels. Deposit results guide closer inspection and do not establish chemical composition. The sample fragments and exposures are synthetic.

FrescoBench builds on the local CoreSection capture, registration, analysis and evidence libraries. Its wall-fragment imagery and deposit analysis ship with the application.
