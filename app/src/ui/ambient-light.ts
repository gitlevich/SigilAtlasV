/**
 * Ambient light tracking — the @Lighting spec's light-follows-attention
 * coupling, realised as the page's ambient tint.
 *
 * Every frame: sample a small window around the @POV centre, look up each
 * sampled image's [brightness, color_temperature] from state.tonal, average
 * with a Gaussian falloff toward the edge of the window, smooth over time
 * with an exponential moving average, and write the result into two CSS
 * custom properties on the document root.
 *
 * CSS reads --ambient-luma (0..1) and --ambient-warmth (-1..1 cool→warm)
 * to drive the body background and any overlay. Attention is literally
 * light: aiming the @POV at dark images darkens the room.
 *
 * Per spec `Explore/Compose/invariant-light-follows-attention`.
 */

import { state } from "../state";
import { isSpaceLikeLayout } from "../types";
import type { AnyLayout } from "../types";
import { imageAtWorld } from "./controls";
import type { TorusViewport } from "../renderer/torus-viewport";

let viewport: TorusViewport | null = null;

/** Wire the viewport so every tick also updates its WebGL clear colour.
 *  Without this, only the DOM body background tracks and the viewport
 *  itself stays the static dark fill. */
export function setAmbientViewport(v: TorusViewport): void {
  viewport = v;
}

// --- sampling parameters ---

// Number of sample points along each axis of the window. 5x5 = 25 samples.
// Cheap and smooth enough; increase for finer tracking if needed.
const SAMPLE_RADIUS = 2;

// Exponential moving average time constant. Tonal values approach the new
// target at exp(-dt/tau). ~350ms gives a felt "settling" response to camera
// moves without making dark→bright feel laggy.
const TAU_MS = 350;

// Neutral fallback when no samples are available (empty slice, no tonal data).
const NEUTRAL_LUMA = 0.20;
const NEUTRAL_WARMTH = 0.0;

// --- internal smoothed state ---

let smoothedLuma = NEUTRAL_LUMA;
let smoothedWarmth = NEUTRAL_WARMTH;
let lastTick = 0;

function cellSize(layout: AnyLayout): number {
  if (isSpaceLikeLayout(layout)) return layout.cell_size;
  return Math.max(1, layout.strip_height);
}

function sampleWindow(layout: AnyLayout, cx: number, cy: number): { luma: number; warmth: number } {
  const step = cellSize(layout);
  let wsum = 0;
  let lsum = 0;
  let tsum = 0;
  const sigma = SAMPLE_RADIUS; // in cell units; controls falloff
  const twoSigma2 = 2 * sigma * sigma;

  for (let dy = -SAMPLE_RADIUS; dy <= SAMPLE_RADIUS; dy++) {
    for (let dx = -SAMPLE_RADIUS; dx <= SAMPLE_RADIUS; dx++) {
      const weight = Math.exp(-(dx * dx + dy * dy) / twoSigma2);
      const id = imageAtWorld(layout, cx + dx * step, cy + dy * step);
      if (!id) continue;
      const tonal = state.tonal[id];
      if (!tonal) continue;
      wsum += weight;
      lsum += tonal[0] * weight;
      tsum += tonal[1] * weight;
    }
  }

  if (wsum === 0) return { luma: NEUTRAL_LUMA, warmth: NEUTRAL_WARMTH };
  return { luma: lsum / wsum, warmth: tsum / wsum };
}

/** Call once per frame. Samples the window around state.pov, updates the
 *  smoothed luma/warmth, writes them to CSS custom properties on <html>. */
export function tickAmbientLight(now: number): void {
  const layout = state.layout;
  const target = layout
    ? sampleWindow(layout, state.pov.x, state.pov.y)
    : { luma: NEUTRAL_LUMA, warmth: NEUTRAL_WARMTH };

  // Exponential moving average: approach target with time constant TAU_MS.
  const dt = lastTick === 0 ? 0 : now - lastTick;
  lastTick = now;
  const alpha = dt > 0 ? 1 - Math.exp(-dt / TAU_MS) : 1;
  smoothedLuma += (target.luma - smoothedLuma) * alpha;
  smoothedWarmth += (target.warmth - smoothedWarmth) * alpha;

  const root = document.documentElement;
  root.style.setProperty("--ambient-luma", smoothedLuma.toFixed(3));
  root.style.setProperty("--ambient-warmth", smoothedWarmth.toFixed(3));

  if (viewport) viewport.setAmbientTint(smoothedLuma, smoothedWarmth);
}
