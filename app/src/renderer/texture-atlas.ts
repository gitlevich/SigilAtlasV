/**
 * Texture atlas for thumbnail images.
 *
 * Packs thumbnails into GPU textures using a shelf-based bin packer.
 * Each image is stored at its natural aspect ratio — no stretching,
 * no fixed grid, no black padding.
 *
 * Concurrency-limited: at most MAX_CONCURRENT fetches in flight.
 * Excess requests queue and drain as in-flight ones complete.
 *
 * Invariants enforced:
 * - !gapless: every pixel of every quad is covered by image content
 * - !undistorted: images maintain their original aspect ratio
 */

const ATLAS_SIZE = 4096;
const MAX_THUMB_H = 96;
const MAX_CONCURRENT = 12;

export interface UVRect {
  u0: number;
  v0: number;
  u1: number;
  v1: number;
}

interface Shelf {
  y: number;
  height: number;
  cursorX: number;
}

interface AtlasPage {
  texture: WebGLTexture;
  shelves: Shelf[];
  nextShelfY: number;
}

export class TextureAtlas {
  private gl: WebGLRenderingContext;
  private pages: AtlasPage[] = [];
  private uvMap: Map<string, { page: number; uv: UVRect }> = new Map();
  private loadingSet: Set<string> = new Set();

  // Concurrency control
  private inFlight = 0;
  private queue: Array<{ id: string; url: string }> = [];

  placeholderUV: UVRect = { u0: 0, v0: 0, u1: 0, v1: 0 };
  placeholderPage = 0;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this._createPage();
    this._initPlaceholder();
  }

  private _createPage(): AtlasPage {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // Allocate without filling — dark gray placeholder is the 1x1 init pixel
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      ATLAS_SIZE, ATLAS_SIZE, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const page: AtlasPage = { texture: tex, shelves: [], nextShelfY: 0 };
    this.pages.push(page);
    return page;
  }

  private _initPlaceholder(): void {
    const gl = this.gl;
    const pixel = new Uint8Array([80, 80, 90, 255]);
    gl.bindTexture(gl.TEXTURE_2D, this.pages[0].texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    this.placeholderUV = { u0: 0, v0: 0, u1: 1 / ATLAS_SIZE, v1: 1 / ATLAS_SIZE };
  }

  private _allocate(w: number, h: number): { page: number; x: number; y: number } | null {
    for (let p = 0; p < this.pages.length; p++) {
      const result = this._allocateOnPage(p, w, h);
      if (result) return { page: p, ...result };
    }
    const newPage = this._createPage();
    const p = this.pages.length - 1;
    return { page: p, ...this._allocateOnPage(p, w, h)! };
  }

  private _allocateOnPage(pageIdx: number, w: number, h: number): { x: number; y: number } | null {
    const page = this.pages[pageIdx];
    for (const shelf of page.shelves) {
      if (h <= shelf.height && shelf.cursorX + w <= ATLAS_SIZE) {
        const x = shelf.cursorX;
        shelf.cursorX += w;
        return { x, y: shelf.y };
      }
    }
    if (page.nextShelfY + h <= ATLAS_SIZE) {
      const shelf: Shelf = { y: page.nextShelfY, height: h, cursorX: w };
      page.shelves.push(shelf);
      const result = { x: 0, y: page.nextShelfY };
      page.nextShelfY += h;
      return result;
    }
    return null;
  }

  getUV(imageId: string): { page: number; uv: UVRect } {
    return this.uvMap.get(imageId) ?? { page: this.placeholderPage, uv: this.placeholderUV };
  }

  isLoaded(imageId: string): boolean {
    return this.uvMap.has(imageId) || this.loadingSet.has(imageId);
  }

  loadThumbnail(imageId: string, url: string): void {
    if (this.uvMap.has(imageId) || this.loadingSet.has(imageId)) return;
    this.loadingSet.add(imageId);

    if (this.inFlight < MAX_CONCURRENT) {
      this._startFetch(imageId, url);
    } else {
      this.queue.push({ id: imageId, url });
    }
  }

  private _startFetch(imageId: string, url: string): void {
    this.inFlight++;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      this._onLoaded(imageId, img);
      this._drainQueue();
    };
    img.onerror = () => {
      this.loadingSet.delete(imageId);
      this.inFlight--;
      this._drainQueue();
    };
    img.src = url;
  }

  private _onLoaded(imageId: string, img: HTMLImageElement): void {
    this.loadingSet.delete(imageId);
    this.inFlight--;

    const aspect = img.naturalWidth / img.naturalHeight;
    const drawH = Math.min(img.naturalHeight, MAX_THUMB_H);
    const drawW = Math.round(drawH * aspect);

    const slot = this._allocate(drawW, drawH);
    if (!slot) return;

    const canvas = document.createElement("canvas");
    canvas.width = drawW;
    canvas.height = drawH;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0, drawW, drawH);

    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.pages[slot.page].texture);
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0,
      slot.x, slot.y,
      gl.RGBA, gl.UNSIGNED_BYTE,
      canvas,
    );

    const uv: UVRect = {
      u0: slot.x / ATLAS_SIZE,
      v0: (slot.y + drawH) / ATLAS_SIZE,
      u1: (slot.x + drawW) / ATLAS_SIZE,
      v1: slot.y / ATLAS_SIZE,
    };
    this.uvMap.set(imageId, { page: slot.page, uv });
  }

  private _drainQueue(): void {
    while (this.inFlight < MAX_CONCURRENT && this.queue.length > 0) {
      const next = this.queue.shift()!;
      // Skip if already loaded while queued
      if (this.uvMap.has(next.id)) {
        this.loadingSet.delete(next.id);
        continue;
      }
      this._startFetch(next.id, next.url);
    }
  }

  pageCount(): number {
    return this.pages.length;
  }

  bindPage(page: number, textureUnit: number): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.pages[page].texture);
  }
}
