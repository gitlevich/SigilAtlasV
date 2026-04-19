/**
 * SigilAtlas main entry point.
 *
 * In Tauri: invokes Rust commands to start the Python sidecar, then connects.
 * In browser dev mode: expects ?port=XXXX URL param for a manually-started sidecar.
 */

import { TorusViewport } from "./renderer/torus-viewport";
import { state, notify, subscribe, initThingsLibrary, refreshCollages } from "./state";
import * as api from "./api";
import { initControls, setViewport, recomputeSliceAndLayout, refreshControls, imageAtWorld, cellCenterAtWorld } from "./ui/controls";
import { initStatusBar } from "./ui/status-bar";
import { initMenu } from "./ui/menu";
import {
  openLightbox,
  closeLightbox,
  isLightboxOpen,
  walkLightbox,
  toggleLightboxMetadata,
  setViewportForLightbox,
  setResumeRenderFn,
} from "./ui/lightbox";
import type { PointOfView } from "./types";
import { isSpaceLikeLayout } from "./types";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

function isTauri(): boolean {
  return typeof window.__TAURI_INTERNALS__ !== "undefined";
}

function showStatus(msg: string, color = "#888"): void {
  const el = document.getElementById("status") ?? (() => {
    const d = document.createElement("div");
    d.id = "status";
    d.style.cssText = `position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);color:${color};font-size:14px;font-family:sans-serif;text-align:center;z-index:100;`;
    document.body.appendChild(d);
    return d;
  })();
  el.style.color = color;
  el.textContent = msg;
}

function hideStatus(): void {
  document.getElementById("status")?.remove();
}

async function startSidecarViaTauri(): Promise<number> {
  const { invoke } = await import("@tauri-apps/api/core");

  showStatus("Starting sidecar...");

  // Resolve paths relative to the app binary location
  // The workspace and python venv are in the project root
  const port = await invoke<number>("start_sidecar", {
    workspace: "/Users/vlad/SigilAtlas/workspace",
    python: "/Users/vlad/SigilAtlas/python/.venv/bin/python",
  });

  return port;
}

async function main(): Promise<void> {
  const t0 = performance.now();
  const mark = (label: string) => console.log(`[startup] ${label}: ${(performance.now() - t0).toFixed(0)}ms`);

  // Status bar subscribes to state — works as soon as state changes, no wiring needed.
  initStatusBar();

  let port: number;

  if (isTauri()) {
    try {
      port = await startSidecarViaTauri();
      mark("sidecar started");
      await initMenu();
      mark("menu");
    } catch (e) {
      showStatus(`Sidecar failed: ${e}`, "#c44");
      return;
    }
  } else {
    const params = new URLSearchParams(window.location.search);
    port = parseInt(params.get("port") ?? "0", 10);
    if (!port) {
      showStatus(
        "No sidecar port. Start the sidecar and add ?port=XXXX to the URL.\n\n" +
        "python -m sigil_atlas.serve --workspace ../workspace",
      );
      return;
    }
  }

  api.setSidecarPort(port);

  // If the sidecar dies (OOM, crash), the next api.* call will hit a network
  // error. Register a revive path so it respawns transparently.
  if (isTauri()) {
    api.setReviveFn(async () => {
      console.warn("[sidecar] dead; respawning");
      return await startSidecarViaTauri();
    });
  }

  // Wait for sidecar health
  let healthy = false;
  for (let i = 0; i < 30; i++) {
    healthy = await api.health();
    if (healthy) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  if (!healthy) {
    showStatus(`Sidecar not responding on port ${port}`, "#c44");
    return;
  }
  mark("sidecar healthy");

  // Init WebGL
  const canvas = document.getElementById("viewport") as HTMLCanvasElement;
  const viewport = new TorusViewport(canvas);
  setViewport(viewport);
  setViewportForLightbox(viewport);
  viewport.setThumbnailBaseUrl(`http://127.0.0.1:${port}`);
  mark("webgl init");

  // Crosshair at the canvas centre — marks the "centred image" that
  // Image > Set Target (\u2318T) and Image > Open in Lightbox (\u2318L)
  // act on. Fired once and on canvas resize (which catches panel folds
  // since the canvas is a flex item).
  const crosshair = document.getElementById("viewport-crosshair");
  const positionCrosshair = () => {
    if (!crosshair) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const w = crosshair.offsetWidth || 18;
    const h = crosshair.offsetHeight || 18;
    crosshair.style.left = `${rect.left + rect.width / 2 - w / 2}px`;
    crosshair.style.top = `${rect.top + rect.height / 2 - h / 2}px`;
    crosshair.style.display = "block";
  };
  positionCrosshair();
  new ResizeObserver(positionCrosshair).observe(canvas);
  window.addEventListener("resize", positionCrosshair);

  // Start loading the baked atlases in the background. First call on a fresh
  // workspace triggers generation (~1min overview, ~3min mid); cached after.
  viewport.loadOverview().then(() => mark("overview atlas ready"));
  viewport.loadMidAtlas().then(() => mark("mid-atlas ready"));

  // Load dimensions and models (with coverage counts for disabling incomplete)
  const [dimensions, modelsRes] = await Promise.all([
    api.getDimensions(),
    api.getModels(),
  ]);
  // Prefer a model whose embeddings are complete. Fall back to the first
  // available if the current default isn't present.
  const completeModels = modelsRes.models.filter(
    (m) => (modelsRes.counts[m] ?? 0) >= modelsRes.total,
  );
  if (completeModels.length > 0 && !completeModels.includes(state.model)) {
    state.model = completeModels[0];
  } else if (modelsRes.models.length > 0 && !modelsRes.models.includes(state.model)) {
    state.model = modelsRes.models[0];
  }
  mark("dimensions + models");

  // Initial slice: entire corpus (null filter = unconstrained)
  const sliceRes = await api.computeSlice({
    filter: null,
    relevance: state.relevance,
    model: state.model,
  });
  state.imageIds = sliceRes.image_ids;
  state.orderValues = sliceRes.order_values || {};
  mark("slice");

  // Initial layout — spacelike by default (gravity field over square-crop cells)
  const layout = await api.computeSpacelike({
    filter: null,
    relevance: state.relevance,
    model: state.model,
    feathering: state.feathering,
    cell_size: state.cellSize,
  });
  state.layout = layout;
  state.torusWidth = layout.torus_width;
  state.torusHeight = layout.torus_height;
  viewport.setLayout(layout);
  mark("layout");

  // Initial framing: ~6 rows of cells in view. The full-torus splash is too
  // far out to read individual cells and too expensive to re-layout when
  // switching to TimeLike — 6 rows is legible and responsive.
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const rowsVisible = 6;
  const desiredZoom = rowsVisible * layout.cell_size * aspect;
  const maxZoom = Math.min(layout.torus_width, layout.torus_height * aspect);
  state.pov = {
    x: layout.torus_width / 2,
    y: layout.torus_height / 2,
    z: Math.min(desiredZoom, maxZoom),
    pitch: 0,
    yaw: 0,
  };

  // Load the ThingsLibrary from the workspace before the panel renders so
  // library pills show up on first paint. Also drains any legacy localStorage
  // entries into the workspace.
  await initThingsLibrary();
  // Load saved collages — non-blocking; UI re-renders when they arrive.
  refreshCollages().catch((e) => console.error("[collages init]", e));
  mark("things library");

  // Init controls
  await initControls(dimensions, modelsRes);
  notify();

  // File-association handling. macOS asks us to open `.sigil` folders by
  // emitting RunEvent::Opened on the Rust side. We listen for the bridged
  // event AND drain any opens that were buffered before this listener
  // attached (cold-launch double-click).
  if (isTauri()) {
    try {
      const { listen } = await import("@tauri-apps/api/event");
      const { invoke } = await import("@tauri-apps/api/core");
      const { loadCollageFromFolder } = await import("./collages");

      await listen<string>("collage-open", (event) => {
        const path = event.payload;
        if (path) loadCollageFromFolder(path).catch((e) => console.error("[open]", e));
      });

      const pending = await invoke<string[]>("drain_pending_opens");
      for (const path of pending) {
        loadCollageFromFolder(path).catch((e) => console.error("[open pending]", e));
      }
    } catch (e) {
      console.error("[file-association]", e);
    }
  }
  mark("controls ready");

  // Refresh viewport when import completes (or errors — images may already be
  // marked completed in the DB even if late pipeline stages like wrapping fail).
  let lastSeenStatus: string | null = null;
  subscribe((s) => {
    const p = s.importProgress;
    if (!p) return;
    const done = p.status === "completed" || p.status === "error";
    if (done && lastSeenStatus !== "completed" && lastSeenStatus !== "error") {
      // Rebuild controls (new dimensions may have appeared) then recompute layout
      refreshControls()
        .then(() => recomputeSliceAndLayout())
        .catch((e) => console.error("[import] refresh failed:", e));
    }
    lastSeenStatus = p.status;
  });

  hideStatus();

  // Camera interaction
  const tickCamera = setupCameraControls(canvas);

  // Render loop — push layer + relief state each frame (cheap)
  let firstFrameLogged = false;
  const getPov = () => {
    tickCamera();
    viewport.setLayers(state.layers);
    viewport.setReliefScale(state.reliefScale);
    if (!firstFrameLogged && state.layout) {
      firstFrameLogged = true;
      requestAnimationFrame(() => mark("first frame"));
    }
    return state.pov;
  };
  viewport.startRenderLoop(getPov);
  // The @Lightbox stops the loop on entry (no attention on the field ⇒ no
  // recomputation) and asks us to restart it on exit. Keeps the loop
  // closure opaque to the lightbox module.
  setResumeRenderFn(() => viewport.startRenderLoop(getPov));
}

/** Set up camera controls. Returns a tick function to call each frame for inertia.
 *
 * Physics model:
 *  - Drag: direct manipulation, content follows pointer 1:1
 *  - Release: velocity from trailing 80ms window, exponential decay (~400ms half-life)
 *  - Grab: kills momentum instantly (finger down = stop)
 *  - Wheel/trackpad: direct pan, no momentum injection
 *  - All decay is time-based (actual dt), not frame-count-based
 */
function setupCameraControls(canvas: HTMLCanvasElement): () => void {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  // Velocity in world-space units per second
  let vx = 0;
  let vy = 0;

  // Trailing window for velocity estimation on release
  const VELOCITY_WINDOW_MS = 80;
  const trail: Array<{ t: number; wx: number; wy: number }> = [];

  // Decay: half-life in seconds. Content loses half its speed every this many seconds.
  const HALF_LIFE = 0.4;
  const DECAY_RATE = Math.log(2) / HALF_LIFE; // continuous decay constant
  const V_EPSILON = 0.5; // world units/s — below this, snap to zero

  let lastFrameT = performance.now();

  function pixelsPerUnit(): number {
    return canvas.clientWidth / state.pov.z;
  }

  // Orbit drag mode: alt/option held on grab
  let orbiting = false;

  // Camera easing target — double-click sets this, tick interpolates.
  let animFrom: { x: number; y: number; z: number } | null = null;
  let animTo: { x: number; y: number; z: number } | null = null;
  let animStart = 0;
  const ANIM_MS = 450;

  // Option+click on a photo makes it a @TargetImage attractor. Deferred by
  // the system double-click window because alt+dblclick zooms out — without
  // the delay both fire on a real double-click and race. Cancelled by the
  // dblclick handler below if a second click arrives in time.
  let pendingTargetId: string | null = null;
  let pendingTargetTimer: number | null = null;
  const DOUBLE_CLICK_MS = 280;

  canvas.addEventListener("click", (e) => {
    if (!e.altKey) return;
    if (!state.layout) return;
    const rect = canvas.getBoundingClientRect();
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const dxPx = (e.clientX - rect.left) - cw / 2;
    const dyPx = (e.clientY - rect.top) - ch / 2;
    const worldPerPx = state.pov.z / cw;
    const worldX = state.pov.x + dxPx * worldPerPx;
    const worldY = state.pov.y - dyPx * worldPerPx;
    const id = imageAtWorld(state.layout, worldX, worldY);
    if (!id) return;
    pendingTargetId = id;
    if (pendingTargetTimer !== null) clearTimeout(pendingTargetTimer);
    pendingTargetTimer = window.setTimeout(() => {
      pendingTargetTimer = null;
      const targetId = pendingTargetId;
      pendingTargetId = null;
      if (!targetId) return;
      state.attractors = state.attractors.filter((a) => a.kind !== "target_image");
      state.attractors.push({ kind: "target_image", ref: targetId });
      recomputeSliceAndLayout({ anchorImageId: targetId })
        .catch((err) => console.error("[attract]", err));
    }, DOUBLE_CLICK_MS);
  });

  canvas.addEventListener("dblclick", (e) => {
    // Cancel any pending option+click target — this is a double-click, not
    // a single, so the deferred target action should not fire.
    if (pendingTargetTimer !== null) {
      clearTimeout(pendingTargetTimer);
      pendingTargetTimer = null;
      pendingTargetId = null;
    }
    const rect = canvas.getBoundingClientRect();
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const dxPx = (e.clientX - rect.left) - cw / 2;
    const dyPx = (e.clientY - rect.top) - ch / 2;
    // World-per-pixel is the same on both axes under square ortho projection.
    // World-per-pixel is the same on both axes under square ortho projection.
    // Y sign is flipped because the renderer's MVP maps larger world-y to
    // NDC +y (top of screen), so clicking above centre means "go to a LARGER
    // world-y", i.e. pov.y - dyPx * wpp (dyPx negative above centre).
    const worldPerPx = state.pov.z / cw;
    const rawX = state.pov.x + dxPx * worldPerPx;
    const rawY = state.pov.y - dyPx * worldPerPx;
    // Snap to the centre of the cell under the click, so the clicked image —
    // not the clicked pixel — becomes the centre of the screen.
    const snap = state.layout ? cellCenterAtWorld(state.layout, rawX, rawY) : null;
    const targetX = snap?.x ?? rawX;
    const targetY = snap?.y ?? rawY;

    // @Lightbox entry: if the @pointOfView already shows three rows or fewer,
    // the next zoom-in gesture lifts the clicked @image out of the field at
    // 100% instead of zooming further into the @surface. Alt-double-click
    // (zoom out) is never an entry — only the zoom-in direction enters.
    const rowHeight = state.layout
      ? (isSpaceLikeLayout(state.layout) ? state.layout.cell_size : state.layout.strip_height)
      : state.cellSize;
    const aspect = cw / ch;
    const rowsVisible = rowHeight > 0 ? state.pov.z / (rowHeight * aspect) : Infinity;
    const THRESHOLD_ROWS = 3;
    if (!e.altKey && state.layout && rowsVisible <= THRESHOLD_ROWS) {
      const id = imageAtWorld(state.layout, rawX, rawY);
      if (id) {
        vx = 0; vy = 0;
        trail.length = 0;
        animFrom = null; animTo = null;
        openLightbox(id);
        return;
      }
    }

    // Alt/Option held: zoom out 2x. Plain double-click: zoom in 2x.
    const zoomingOut = e.altKey;
    const factor = zoomingOut ? 2 : 0.5;
    const tw = state.torusWidth || 10000;
    const th = state.torusHeight || 10000;
    const maxZ = Math.min(tw, th * aspect);
    const newZ = Math.max(100, Math.min(maxZ, state.pov.z * factor));

    // Short-wrap the target: if the straight-line path from animFrom to animTo
    // is longer than half the torus, go the other way around. Otherwise the
    // camera sweeps across the full torus when the click is near a seam.
    const tw2 = state.torusWidth || 0;
    const th2 = state.torusHeight || 0;
    let dX = targetX - state.pov.x;
    let dY = targetY - state.pov.y;
    if (tw2 > 0) {
      if (dX > tw2 / 2) dX -= tw2;
      else if (dX < -tw2 / 2) dX += tw2;
    }
    if (th2 > 0) {
      if (dY > th2 / 2) dY -= th2;
      else if (dY < -th2 / 2) dY += th2;
    }
    animFrom = { x: state.pov.x, y: state.pov.y, z: state.pov.z };
    animTo = { x: state.pov.x + dX, y: state.pov.y + dY, z: newZ };
    animStart = performance.now();
    // Kill any residual inertia velocity that would otherwise fling the
    // camera after the animation completes — micro-motions between the two
    // clicks of a double-click leave vx/vy non-zero.
    vx = 0; vy = 0;
    trail.length = 0;
  });

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    orbiting = e.altKey;
    lastX = e.clientX;
    lastY = e.clientY;
    // Grab kills momentum — this is what users expect
    vx = 0;
    vy = 0;
    trail.length = 0;
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const now = performance.now();
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (orbiting) {
      // Orbit: horizontal drag rotates yaw, vertical drag tilts pitch.
      // Only meaningful when relief is on, but always live.
      const YAW_PER_PX = 0.006;
      const PITCH_PER_PX = 0.006;
      state.pov.yaw += dx * YAW_PER_PX;
      state.pov.pitch = Math.max(0, Math.min(Math.PI / 2 - 0.02, state.pov.pitch + dy * PITCH_PER_PX));
      return;
    }

    const ppu = pixelsPerUnit();
    const worldDx = -dx / ppu;
    const worldDy = dy / ppu;

    // Direct manipulation: content follows pointer 1:1
    state.pov.x += worldDx;
    state.pov.y += worldDy;
    wrapPov();

    // Record in trailing window for velocity estimation
    trail.push({ t: now, wx: worldDx, wy: worldDy });
    // Evict old samples
    const cutoff = now - VELOCITY_WINDOW_MS * 2;
    while (trail.length > 0 && trail[0].t < cutoff) trail.shift();
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    if (orbiting) { orbiting = false; return; }

    // Estimate release velocity from trailing window
    const now = performance.now();
    const cutoff = now - VELOCITY_WINDOW_MS;
    let totalWx = 0, totalWy = 0, totalT = 0;
    for (const s of trail) {
      if (s.t >= cutoff) {
        totalWx += s.wx;
        totalWy += s.wy;
      }
    }
    // Time span of the samples in the window
    const recent = trail.filter(s => s.t >= cutoff);
    if (recent.length >= 2) {
      totalT = (recent[recent.length - 1].t - recent[0].t) / 1000;
    }
    if (totalT > 0.005) {
      vx = totalWx / totalT;
      vy = totalWy / totalT;
    } else {
      vx = 0;
      vy = 0;
    }
    trail.length = 0;
  });

  function applyZoom(factor: number): void {
    const aspect = canvas.clientWidth / canvas.clientHeight;
    const tw = state.torusWidth || 10000;
    const th = state.torusHeight || 10000;
    const maxZoom = Math.min(tw, th * aspect);
    state.pov.z = Math.max(50, Math.min(maxZoom, state.pov.z * factor));
  }

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (e.ctrlKey) {
      // Pinch-to-zoom (trackpad sends wheel+ctrlKey for pinch)
      applyZoom(1 + e.deltaY * 0.005);
    } else {
      // Two-finger swipe → direct pan, no momentum
      const ppu = pixelsPerUnit();
      state.pov.x -= e.deltaX / ppu;
      state.pov.y += e.deltaY / ppu;
      wrapPov();
    }
  }, { passive: false });

  // Cmd+= / Cmd+- / Cmd+0 / Cmd+\\ are owned by the View menu accelerators
  // when running in Tauri (native menu). The keydown handler here would
  // double-fire and is intentionally absent.

  // @Lightbox keyboard: Escape exits, arrows walk the suspended
  // @arrangement's lattice, Cmd-I toggles the metadata overlay. These
  // handlers are registered before the others so they consume the keystroke
  // when the @Lightbox is open; the zoom and target-image handlers early-out
  // via isLightboxOpen().
  window.addEventListener("keydown", (e) => {
    if (!isLightboxOpen()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      closeLightbox();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && (e.key === "i" || e.key === "I")) {
      e.preventDefault();
      toggleLightboxMetadata();
      return;
    }
    const dirMap: Record<string, "up" | "down" | "left" | "right"> = {
      ArrowUp: "up",
      ArrowDown: "down",
      ArrowLeft: "left",
      ArrowRight: "right",
    };
    const dir = dirMap[e.key];
    if (dir) {
      e.preventDefault();
      walkLightbox(dir);
    }
  });

  // SpaceLike arrow-key navigation — step the @POV one cell at a time on
  // the @torus surface. Up/down move on Y (down = increasing y, visually
  // "further along" the strip axis); left/right move on X. The torus
  // wraps, so stepping past an edge returns from the opposite side.
  // Shift+Arrow steps a full screen's worth; holding a key auto-repeats
  // at the OS rate. Skipped while the @Lightbox is open (it owns arrows
  // for lattice walking) or while typing into an input.
  window.addEventListener("keydown", (e) => {
    if (isLightboxOpen()) return;
    if (state.mode !== "spacelike") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (t) {
      const tag = t.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || t.isContentEditable) return;
    }
    let dx = 0, dy = 0;
    switch (e.key) {
      case "ArrowUp": dy = -1; break;
      case "ArrowDown": dy = 1; break;
      case "ArrowLeft": dx = -1; break;
      case "ArrowRight": dx = 1; break;
      default: return;
    }
    e.preventDefault();
    const step = state.cellSize || 100;
    const big = e.shiftKey ? Math.max(1, Math.floor(state.pov.z / step)) : 1;
    state.pov.x += dx * step * big;
    state.pov.y += dy * step * big;
    const tw = state.torusWidth || 1;
    const th = state.torusHeight || 1;
    state.pov.x = ((state.pov.x % tw) + tw) % tw;
    state.pov.y = ((state.pov.y % th) + th) % th;
  });

  // Escape releases an active TargetImage. The pill is visible in
  // AttractorControl, but the keyboard remains a handy quick release.
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (isLightboxOpen()) return;
    const before = state.attractors.length;
    state.attractors = state.attractors.filter((a) => a.kind !== "target_image");
    if (state.attractors.length !== before) {
      recomputeSliceAndLayout().catch((err) => console.error("[release-target]", err));
    }
  });

  function wrapPov(): void {
    const tw = state.torusWidth || 1;
    const th = state.torusHeight || 1;
    state.pov.x = ((state.pov.x % tw) + tw) % tw;
    state.pov.y = ((state.pov.y % th) + th) % th;
  }

  // Per-frame inertia tick — time-based exponential decay
  return () => {
    const now = performance.now();
    const dt = Math.min((now - lastFrameT) / 1000, 0.05); // cap at 50ms to avoid jumps
    lastFrameT = now;

    // Double-click zoom animation takes priority over inertia.
    if (animFrom && animTo) {
      const t = Math.min(1, (now - animStart) / ANIM_MS);
      const eased = 0.5 - 0.5 * Math.cos(t * Math.PI);
      state.pov.x = animFrom.x + (animTo.x - animFrom.x) * eased;
      state.pov.y = animFrom.y + (animTo.y - animFrom.y) * eased;
      state.pov.z = animFrom.z + (animTo.z - animFrom.z) * eased;
      wrapPov();
      if (t >= 1) { animFrom = null; animTo = null; }
      return;
    }

    if (dragging) return;
    if (Math.abs(vx) < V_EPSILON && Math.abs(vy) < V_EPSILON) {
      vx = 0;
      vy = 0;
      return;
    }

    // Apply velocity
    state.pov.x += vx * dt;
    state.pov.y += vy * dt;
    wrapPov();

    // Exponential decay: v *= e^(-rate * dt)
    const decay = Math.exp(-DECAY_RATE * dt);
    vx *= decay;
    vy *= decay;
  };
}

main();
