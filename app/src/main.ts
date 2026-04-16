/**
 * SigilAtlas main entry point.
 *
 * In Tauri: invokes Rust commands to start the Python sidecar, then connects.
 * In browser dev mode: expects ?port=XXXX URL param for a manually-started sidecar.
 */

import { TorusViewport } from "./renderer/torus-viewport";
import { state, notify, subscribe } from "./state";
import * as api from "./api";
import { initControls, setViewport, recomputeSliceAndLayout, refreshControls } from "./ui/controls";
import { initStatusBar } from "./ui/status-bar";
import { initMenu } from "./ui/menu";
import type { PointOfView } from "./types";

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
  viewport.setThumbnailBaseUrl(`http://127.0.0.1:${port}`);
  mark("webgl init");

  // Load dimensions and models
  const [dimensions, models] = await Promise.all([
    api.getDimensions(),
    api.getModels(),
  ]);
  // Auto-select first available model if current default isn't available
  if (models.length > 0 && !models.includes(state.model)) {
    state.model = models[0];
  }
  mark("dimensions + models");

  // Initial slice: entire corpus
  const sliceRes = await api.computeSlice({
    range_filters: [],
    proximity_filters: [],
    contrast_controls: [],
    model: state.model,
    tightness: state.tightness,
  });
  state.imageIds = sliceRes.image_ids;
  state.orderValues = sliceRes.order_values || {};
  mark("slice");

  // Initial layout — spacelike by default
  const layout = await api.computeLayout({
    image_ids: state.imageIds,
    axes: null,
    tightness: state.tightness,
    model: state.model,
    strip_height: state.stripHeight,
  });
  state.layout = layout;
  state.torusWidth = layout.torus_width;
  state.torusHeight = layout.torus_height;
  viewport.setLayout(layout);
  mark("layout");

  // Center camera
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const maxZoom = Math.min(layout.torus_width, layout.torus_height * aspect);
  const desiredZoom = 8 * layout.strip_height * aspect;
  state.pov = { x: layout.torus_width / 2, y: layout.torus_height / 2, z: Math.min(desiredZoom, maxZoom) };

  // Init controls
  await initControls(dimensions, models);
  notify();
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

  // Render loop
  let firstFrameLogged = false;
  viewport.startRenderLoop(() => {
    tickCamera();
    if (!firstFrameLogged && state.layout && state.layout.strips.length > 0) {
      firstFrameLogged = true;
      requestAnimationFrame(() => mark("first frame"));
    }
    return state.pov;
  });
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

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
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

  // Cmd+/Cmd- keyboard zoom
  window.addEventListener("keydown", (e) => {
    if (!(e.metaKey || e.ctrlKey)) return;
    if (e.key === "=" || e.key === "+") {
      e.preventDefault();
      applyZoom(0.85); // zoom in
    } else if (e.key === "-") {
      e.preventDefault();
      applyZoom(1.18); // zoom out
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
