/**
 * Lightbox — one image inhabited in isolation.
 *
 * Per sigil_atlas.sigil/Explore/Lightbox: the @slice, @arrangement, and
 * @control are suspended while the Lightbox is open. There is only one
 * attention in this app, and when it is here it is not on the field. Exit
 * returns to exactly the @Frame the user left.
 *
 * Entry is triggered from main.ts when a zoom-in gesture would pass the
 * three-row threshold on the @pointOfView. The Lightbox owns the keyboard
 * and pointer while open: Escape exits; arrow keys walk the suspended
 * @arrangement's lattice; Cmd-I toggles the metadata overlay.
 */

import { state, notify } from "../state";
import { imageNeighbor } from "./controls";
import * as api from "../api";
import type { ImageMetadata } from "../api";
import type { TorusViewport } from "../renderer/torus-viewport";

let rootEl: HTMLDivElement | null = null;
let imgEl: HTMLImageElement | null = null;
let metaEl: HTMLDivElement | null = null;
let viewport: TorusViewport | null = null;

// Metadata cache — the info panel for a walked-to image should appear
// instantly if we've seen it before. Cleared when the lightbox closes.
const metaCache = new Map<string, ImageMetadata | null>();

// Tracks the token for the in-flight full-resolution swap. If the user walks
// to a different image before the original finishes loading, the older load
// is discarded so a late arrival can't overwrite the current image.
let sourceToken = 0;

export function setViewportForLightbox(vp: TorusViewport): void {
  viewport = vp;
}

function ensureDom(): void {
  if (rootEl) return;
  rootEl = document.createElement("div");
  rootEl.id = "lightbox";
  rootEl.hidden = true;

  imgEl = document.createElement("img");
  imgEl.className = "lightbox-img";
  imgEl.alt = "";

  metaEl = document.createElement("div");
  metaEl.className = "lightbox-meta";
  metaEl.hidden = true;

  rootEl.appendChild(imgEl);
  rootEl.appendChild(metaEl);
  document.body.appendChild(rootEl);
}

export function isLightboxOpen(): boolean {
  return state.lightbox.imageId !== null;
}

export function openLightbox(imageId: string): void {
  ensureDom();
  // Snapshot the @Frame so !field-frozen holds: exit restores pan/zoom
  // exactly. The caller (dblclick) has already killed any in-flight
  // camera animation; we don't touch it further.
  state.lightbox.entryPov = { ...state.pov };
  state.lightbox.imageId = imageId;
  renderImage(imageId);
  applyMetadataVisibility();
  rootEl!.hidden = false;
  viewport?.stopRenderLoop();
  notify();
}

export function closeLightbox(): void {
  if (!isLightboxOpen()) return;
  sourceToken++; // invalidate any pending source swap
  state.lightbox.imageId = null;
  state.lightbox.entryPov = null;
  metaCache.clear();
  if (rootEl) rootEl.hidden = true;
  if (imgEl) imgEl.src = "";
  if (metaEl) { metaEl.innerHTML = ""; metaEl.hidden = true; }
  // Resume the render loop with the original tick — main.ts re-attaches it.
  resumeRender?.();
  notify();
}

// main.ts owns the render tick (setupCameraControls' ticker). The lightbox
// stops the loop on entry and asks main to restart it on exit via this
// callback — keeps the tick closure opaque to this module.
let resumeRender: (() => void) | null = null;
export function setResumeRenderFn(fn: () => void): void {
  resumeRender = fn;
}

export function walkLightbox(dir: "up" | "down" | "left" | "right"): void {
  if (!isLightboxOpen() || !state.layout) return;
  const here = state.lightbox.imageId!;
  const next = imageNeighbor(state.layout, here, dir);
  if (!next || next === here) return;
  state.lightbox.imageId = next;
  renderImage(next);
  notify();
}

export function toggleLightboxMetadata(): void {
  state.lightbox.showMetadata = !state.lightbox.showMetadata;
  applyMetadataVisibility();
  notify();
}

function renderImage(imageId: string): void {
  if (!imgEl) return;
  // Cascade: preview → thumbnail (guaranteed floor) → source. The thumbnail
  // is 256px but always exists for completed images, so the user never sees
  // a broken-image icon. Preview and source each upgrade in when available.
  const previewUrl = api.imagePreviewUrl(imageId);
  const thumbnailUrl = api.imageThumbnailUrl(imageId);
  const sourceUrl = api.imageSourceUrl(imageId);

  imgEl.onerror = () => {
    // Preview missing — drop to thumbnail. Swap the handler so a thumbnail
    // failure doesn't loop; a completed image should always have one, but
    // if not, leave the element empty rather than cycling.
    if (state.lightbox.imageId !== imageId) return;
    imgEl!.onerror = null;
    imgEl!.src = thumbnailUrl;
  };
  imgEl.src = previewUrl;

  const token = ++sourceToken;
  const probe = new Image();
  probe.onload = () => {
    if (token !== sourceToken) return;
    if (state.lightbox.imageId !== imageId) return;
    imgEl!.onerror = null; // source superseded preview; don't fall back on its error
    imgEl!.src = sourceUrl;
  };
  probe.onerror = () => {
    // Source unavailable (sidecar returned 204 or the file moved). Keep
    // whatever the cascade has already landed on.
  };
  probe.src = sourceUrl;

  renderMetadata(imageId);
}

async function renderMetadata(imageId: string): Promise<void> {
  if (!metaEl) return;
  if (!state.lightbox.showMetadata) return; // panel hidden, defer work
  if (metaCache.has(imageId)) {
    paintMetadata(metaCache.get(imageId) ?? null);
    return;
  }
  metaEl.textContent = "Loading…";
  const info = await api.getImageMetadata(imageId);
  metaCache.set(imageId, info);
  if (state.lightbox.imageId !== imageId) return; // walked away
  paintMetadata(info);
}

function applyMetadataVisibility(): void {
  if (!metaEl) return;
  metaEl.hidden = !state.lightbox.showMetadata;
  if (state.lightbox.showMetadata && state.lightbox.imageId) {
    renderMetadata(state.lightbox.imageId);
  }
}

function paintMetadata(info: ImageMetadata | null): void {
  if (!metaEl) return;
  if (!info) {
    metaEl.textContent = "No metadata.";
    return;
  }
  const rows: Array<[string, string]> = [];
  if (info.source_path) rows.push(["Path", info.source_path]);
  if (info.capture_date) rows.push(["Captured", formatDate(info.capture_date)]);
  if (info.pixel_width && info.pixel_height) {
    rows.push(["Dimensions", `${info.pixel_width} × ${info.pixel_height}`]);
  }
  if (info.camera_model) rows.push(["Camera", info.camera_model]);
  if (info.lens_model) rows.push(["Lens", info.lens_model]);
  const exposure: string[] = [];
  if (info.focal_length) exposure.push(`${info.focal_length.toFixed(0)}mm`);
  if (info.aperture) exposure.push(`f/${info.aperture.toFixed(1)}`);
  if (info.shutter_speed) exposure.push(formatShutter(info.shutter_speed));
  if (info.iso) exposure.push(`ISO ${info.iso}`);
  if (exposure.length > 0) rows.push(["Exposure", exposure.join("  ")]);
  if (info.gps_latitude !== null && info.gps_longitude !== null) {
    rows.push(["Location", `${info.gps_latitude.toFixed(5)}, ${info.gps_longitude.toFixed(5)}`]);
  }
  metaEl.innerHTML = rows
    .map(([k, v]) => `<div class="lightbox-meta-row"><span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(v)}</span></div>`)
    .join("");
}

function formatDate(ts: number): string {
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toLocaleString();
}

function formatShutter(sec: number): string {
  if (sec >= 1) return `${sec.toFixed(1)}s`;
  const denom = Math.round(1 / sec);
  return `1/${denom}s`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]!));
}
