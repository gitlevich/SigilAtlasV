/**
 * Overview atlas — a single precomputed texture containing a tiny tile per
 * image. Baked at the sidecar, downloaded once at startup, uploaded as one
 * WebGL texture. Used as the fallback layer: whenever an image has no entry
 * in the regular atlas (not yet fetched), the renderer samples its tile here.
 *
 * Effect: the entire field appears coherent from the first frame, with higher
 * resolution swapping in as individual thumbnails load.
 */

import type { UVRect } from "./texture-atlas";

interface Mapping {
  [id: string]: [number, number];
}

interface IndexResponse {
  tile_size: number;
  cols: number;
  rows: number;
  atlas_width: number;
  atlas_height: number;
  mapping: Mapping;
}

export class OverviewAtlas {
  private gl: WebGLRenderingContext;
  private texture: WebGLTexture | null = null;
  private uvMap: Map<string, UVRect> = new Map();
  private ready = false;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
  }

  isReady(): boolean {
    return this.ready;
  }

  getUV(id: string): UVRect | null {
    return this.uvMap.get(id) ?? null;
  }

  bind(textureUnit: number): void {
    if (!this.texture) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
  }

  async load(baseUrl: string): Promise<void> {
    const indexRes = await fetch(`${baseUrl}/overview/index`);
    if (!indexRes.ok) throw new Error(`/overview/index: ${indexRes.status}`);
    const idx: IndexResponse = await indexRes.json();

    // Precompute per-id UV rect in [0, 1]^2.
    // Atlas stored bottom-to-top like texture-atlas.ts (v flipped).
    const aw = idx.atlas_width, ah = idx.atlas_height;
    for (const [id, [col, row]] of Object.entries(idx.mapping)) {
      const x0 = col * idx.tile_size;
      const y0 = row * idx.tile_size;
      const x1 = x0 + idx.tile_size;
      const y1 = y0 + idx.tile_size;
      this.uvMap.set(id, {
        u0: x0 / aw,
        v0: y1 / ah,
        u1: x1 / aw,
        v1: y0 / ah,
      });
    }

    // Fetch and decode the PNG, then upload as a texture.
    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("Failed to load overview atlas image"));
      img.src = `${baseUrl}/overview/atlas`;
    });

    const gl = this.gl;
    this.texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.ready = true;
  }
}
