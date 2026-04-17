/**
 * Mid-atlas — multi-page 32×32 px tiles covering every image. Downloaded
 * once at startup (one PNG per page), cached on disk by the sidecar.
 *
 * Fills the zoom band between the 15px overview (coarse) and per-image 96px
 * atlas (slow to stream under HTTP/1.1 connection caps). Used when a cell
 * is rendered between ~30 and ~80 px on screen.
 */

import type { UVRect } from "./texture-atlas";

interface Mapping {
  [id: string]: [number, number, number]; // [page, col, row]
}

interface IndexResponse {
  tile_size: number;
  cols_per_page: number;
  rows_per_page: number;
  atlas_width: number;
  atlas_height: number;
  pages: number;
  mapping: Mapping;
}

interface PageEntry {
  texture: WebGLTexture;
}

export class MidAtlas {
  private gl: WebGLRenderingContext;
  private pages: PageEntry[] = [];
  private uvMap: Map<string, { page: number; uv: UVRect }> = new Map();
  private ready = false;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
  }

  isReady(): boolean {
    return this.ready;
  }

  getUV(id: string): { page: number; uv: UVRect } | null {
    return this.uvMap.get(id) ?? null;
  }

  bindPage(page: number, textureUnit: number): void {
    const entry = this.pages[page];
    if (!entry) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, entry.texture);
  }

  async load(baseUrl: string): Promise<void> {
    const indexRes = await fetch(`${baseUrl}/midatlas/index`);
    if (!indexRes.ok) throw new Error(`/midatlas/index: ${indexRes.status}`);
    const idx: IndexResponse = await indexRes.json();

    const aw = idx.atlas_width, ah = idx.atlas_height;
    for (const [id, [page, col, row]] of Object.entries(idx.mapping)) {
      const x0 = col * idx.tile_size;
      const y0 = row * idx.tile_size;
      const x1 = x0 + idx.tile_size;
      const y1 = y0 + idx.tile_size;
      this.uvMap.set(id, {
        page,
        uv: { u0: x0 / aw, v0: y1 / ah, u1: x1 / aw, v1: y0 / ah },
      });
    }

    // Fetch and upload each page in parallel.
    const gl = this.gl;
    const loads = Array.from({ length: idx.pages }, async (_, page) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error(`Failed to load mid-atlas page ${page}`));
        img.src = `${baseUrl}/midatlas/page/${page}`;
      });
      const texture = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.pages[page] = { texture };
    });
    await Promise.all(loads);

    this.ready = true;
  }
}
