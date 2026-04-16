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

/** Set up camera controls. Returns a tick function to call each frame for inertia. */
function setupCameraControls(canvas: HTMLCanvasElement): () => void {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let lastT = 0;

  // Velocity in world-space units per second
  let vx = 0;
  let vy = 0;

  const FRICTION = 0.985; // per-frame decay — heavy, long coast (~7s to stop)
  const V_EPSILON = 0.1;  // stop threshold (world units/s)

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    lastT = performance.now();
    // Don't zero velocity — grab adds drag, not a hard stop.
    // The pointer tracking in mousemove will blend new motion
    // into the existing momentum.
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const now = performance.now();
    const dt = Math.max(now - lastT, 1) / 1000;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    lastT = now;

    const pixelsPerUnit = canvas.clientWidth / state.pov.z;
    const worldDx = -dx / pixelsPerUnit;
    const worldDy = dy / pixelsPerUnit;

    state.pov.x += worldDx;
    state.pov.y += worldDy;
    wrapPov();

    // Exponential smoothing avoids jitter on release
    vx = 0.6 * (worldDx / dt) + 0.4 * vx;
    vy = 0.6 * (worldDy / dt) + 0.4 * vy;
  });

  window.addEventListener("mouseup", () => { dragging = false; });

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
      // Two-finger swipe → pan with inertia (heavy, slow)
      // Content follows fingers: positive deltaX = finger moved right = content moves right
      const pixelsPerUnit = canvas.clientWidth / state.pov.z;
      const mass = 0.25; // <1 = heavier feel
      const worldDx = -e.deltaX / pixelsPerUnit * mass;
      const worldDy = e.deltaY / pixelsPerUnit * mass;
      state.pov.x += worldDx;
      state.pov.y += worldDy;
      wrapPov();
      // Feed into velocity so releasing fingers coasts
      vx = vx * 0.5 + worldDx * 60 * 0.5;
      vy = vy * 0.5 + worldDy * 60 * 0.5;
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

  // Return the per-frame inertia tick
  return () => {
    if (dragging) return;
    if (Math.abs(vx) < V_EPSILON && Math.abs(vy) < V_EPSILON) {
      vx = 0;
      vy = 0;
      return;
    }
    state.pov.x += vx / 60;
    state.pov.y += vy / 60;
    wrapPov();
    vx *= FRICTION;
    vy *= FRICTION;
  };
}

main();
