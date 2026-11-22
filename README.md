# FrescoBench

FrescoBench is a local workbench for conservators inspecting photographed wall painting fragments. It retains original exposures, registers views on a measured plane and saves the support behind each reading.

## Start

Run `docker compose up --build --wait` and open `http://127.0.0.1:19989`. Choose another local port with `FRESCOBENCH_PORT`. A named volume stores captures and measurements. With Python 3.11 and the pinned dependencies installed, `./start.sh` starts the same workbench.

## Inspect a fragment

1. Load the four synthetic sample fragments or add photos in **Captures**. Mark the four survey corners, enter the plane width and height in millimeters, and register each alternate exposure against the corresponding corners. In **Sampling regions**, distinguish inspected areas, material references and exclusions.
2. In **Measure**, use **Salt bloom candidates** on the painted face of *Ultramarine · W12*. It uses an intact pigment control to find connected, light, low-chroma deposits and marks candidates reaching a support boundary. **Loss margin width** measures bare plaster on each side of a marked loss strip. **Paint coverage** separates bare, painted and ambiguous areas. **Plaster pore candidates** distinguishes closed dark footprints from truncated ones. Each measurement retains a map and downloadable readings.
3. Use **Surface observations** to review individual marks, **Polarized glare** and **Raking light comparison** for photograph pairs, **Pigment texture spacing** and **Fine-line network** for close inspection, and **Geometry** for physical traces and areas. **Fragment studies** compares retained measurements. **Evidence** builds a ZIP of selected measurements and optional originals. **Received evidence** verifies the declared data in an opened evidence ZIP.

### Continue spots between two photographs

Review the spots of the first *Surface observations* run, then run *Surface observations* again on another registered photograph of the same specimen. In the second run, **Continue from an earlier run** picks the earlier run and asks for your two limits: the largest accepted center offset in registered-plane pixels and the smallest accepted footprint overlap (intersection over union of the two outlined footprints). A pair may continue only where both runs photograph the position, which means its center lies on unclipped opaque support in both registered captures.

The relation is one-to-one: each earlier spot continues at most one later spot and each later spot continues at most one earlier spot. It is chosen in this order: keep as many continuations as possible, then keep as many earlier spots that you had confirmed as possible, then make the summed footprint overlap as high as possible, then make the summed center offset as small as possible. The chosen order and the two limits are saved with the relation and listed on screen, and only this one relation is kept.

Every earlier spot that does not continue stays in the archive with either *Not photographed in the other view* or *No qualifying continuation*; none is recorded as removed. Both runs keep their own review decisions.

Open any later spot's **Note** dialog to see the earlier run that it continues, the earlier decision and notes, and a close image of that earlier spot. **Change** on an earlier spot row moves only that spot: you can assign it to another qualifying later spot or clear its continuation, with an optional reason. A manual move never rewrites the other saved relations; an earlier spot you clear stays cleared until you change it back or rebuild the relation. Earlier and later reviews are untouched. The change history and a *Continuation CSV* of every earlier spot and its outcome are retained with the run and in pinned evidence.

The same earlier run can feed more than one later photograph, for example a second capture and a raking-light capture, and **Feeds later runs** on the earlier run links to each of them. A chain may not loop back on itself. Continuation needs both runs on the same measured plane with unchanged capture alignment; if either changes, the saved relation is shown with a warning and must be rebuilt.

Inputs are single-frame PNG, JPEG, WebP or TIFF up to 24 MiB and 12 megapixels. A specimen holds at most 10 captures and 24 regions. Registration and distances are planar. The longest side of the registered sampling plane is limited to 1,800 pixels. Deposit results guide closer inspection and do not establish chemical composition. The sample fragments and exposures are synthetic.

FrescoBench builds on the local CoreSection capture, registration, analysis and evidence libraries. Its wall-fragment imagery and deposit analysis ship with the application.
