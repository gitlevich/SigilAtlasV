---
status: idea
---

# Screen

The application window. A large area dominated by the @frame, with a foldable @rightPanel, @leftPanel, and @statusBar.

The screen is !dedicated-to-images — the @frame takes maximum space. Panels fold away to give full screen to the @frame.

Its contents at any one moment are determined by the current @PointOfView and @mode.

## Layout

- @frame: center, fills available space. Shows the @torus surface at the current @PointOfView.
- @rightPanel: foldable. Contains @sigilControls for defining @RelevanceFilter, switching @mode, adjusting @timescale.
- @leftPanel: foldable. Image details when one is selected.
- @statusBar: bottom edge. Current @mode, @slice size, position info.

## Navigation affordances

- Pan: drag or arrow keys. Moves @PointOfView along the @torus surface.
- Zoom: scroll wheel or pinch. Adjusts distance from surface. Close = one image fills @frame. Far = many @strips visible.
- Click image: select it. Opens @leftPanel with details.
- Double-click image: center on it. "More like this" in @tastelike mode.
