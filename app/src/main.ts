/**
 * SigilAtlas main entry point.
 *
 * In Tauri: invokes Rust commands to start the Python sidecar, then connects.
 * In browser dev mode: expects ?port=XXXX URL param for a manually-started sidecar.
 */

import { TorusViewport } from "./renderer/torus-viewport";
import { state, notify } from "./state";
import * as api from "./api";
import { initControls, setViewport } from "./ui/controls";
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
  let port: number;

  if (isTauri()) {
    try {
      port = await startSidecarViaTauri();
    } catch (e) {
      showStatus(`Sidecar failed: ${e}`, "#c44");
      return;
    }
  } else {
    // Browser dev mode: get port from URL
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
  showStatus("Connecting to sidecar...");

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

  showStatus("Loading corpus...");

  // Init WebGL
  const canvas = document.getElementById("viewport") as HTMLCanvasElement;
  const viewport = new TorusViewport(canvas);
  setViewport(viewport);
  viewport.setThumbnailBaseUrl(`http://127.0.0.1:${port}`);

  // Load dimensions and models
  const [dimensions, models] = await Promise.all([
    api.getDimensions(),
    api.getModels(),
  ]);

  showStatus("Computing layout...");

  // Initial slice: entire corpus
  const sliceRes = await api.computeSlice({
    range_filters: [],
    proximity_filters: [],
    model: state.model,
  });
  state.imageIds = sliceRes.image_ids;

  // Initial layout — torus dimensions derived from content (!gapless)
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

  // Center camera, zoom to show ~8 strips worth of content
  const visibleStrips = 8;
  const initialZoom = visibleStrips * layout.strip_height * (canvas.clientWidth / canvas.clientHeight);
  state.pov = { x: layout.torus_width / 2, y: layout.torus_height / 2, z: initialZoom };

  // Init controls
  await initControls(dimensions, models);
  notify();

  hideStatus();

  // Camera interaction
  setupCameraControls(canvas, state.pov, layout.torus_width, layout.torus_height);

  // Render loop
  viewport.startRenderLoop(() => state.pov);
}

function setupCameraControls(
  canvas: HTMLCanvasElement,
  pov: PointOfView,
  torusW: number,
  torusH: number,
): void {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    const pixelsPerUnit = canvas.clientWidth / pov.z;
    pov.x -= dx / pixelsPerUnit;
    pov.y += dy / pixelsPerUnit;

    pov.x = ((pov.x % torusW) + torusW) % torusW;
    pov.y = ((pov.y % torusH) + torusH) % torusH;
  });

  window.addEventListener("mouseup", () => { dragging = false; });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = 1 + e.deltaY * 0.001;
    // Max zoom = torus width (one full period visible). The torus wraps,
    // so you always see a full surface — never empty space (!gapless, !endless).
    pov.z = Math.max(50, Math.min(torusW, pov.z * zoomFactor));
  }, { passive: false });
}

main();
