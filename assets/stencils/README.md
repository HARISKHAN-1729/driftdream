# Stencils

Masks for `laminar` targets: the region that should appear to drift while the
background stays still.

Two ways to supply one:

1. **Rasterised text** (no file needed) -- set `text:` in the target config:

       target: {kind: laminar, angle_deg: 180.0, text: DRIFT}

2. **A PNG in this directory** -- white = drifting region, black = background,
   any size (it is used at the working resolution). Load it yourself and pass
   it as `stencil=` to `driftdream.targets.laminar`.

Keep stencils bold and low-frequency. Thin strokes fall below the motion
reader's receptive field and the drift becomes unmeasurable -- see the
resolution finding in the README (peripheral drift is under-resolved below
~128px, and a stencil stroke is effectively its own small image).
