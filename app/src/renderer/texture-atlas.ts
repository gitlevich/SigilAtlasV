/**
 * Texture atlas for thumbnail images.
 *
 * Packs thumbnails into GPU textures using a shelf-based bin packer.
 * Each image is stored at its natural aspect ratio — no stretching,
 * no fixed grid, no black padding.
 *
 * Invariants enforced:
 * - !gapless: every pixel of every quad is covered by image content
 * - !undistorted: images maintain their original aspect ratio
 */

const ATLAS_SIZE = 4096;
const MAX_THUMB_H = 96; // Max height for thumbnails in atlas

export interface UVRect {
  u0: number;
  v0: number;
  u1: number;
  v1: number;
}

/** A horizontal shelf in the atlas that holds images of similar height. */
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
    // Initialize with a dark gray (not black) to make leaks visible during debug
    const pixels = new Uint8Array(ATLAS_SIZE * ATLAS_SIZE * 4);
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i] = 40; pixels[i + 1] = 40; pixels[i + 2] = 48; pixels[i + 3] = 255;
    }
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      ATLAS_SIZE, ATLAS_SIZE, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, pixels,
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

  /**
   * Shelf-based bin packing: find a shelf that fits this image,
   * or create a new shelf. Returns atlas coordinates for the image.
   */
  private _allocate(w: number, h: number): { page: number; x: number; y: number } | null {
    for (let p = 0; p < this.pages.length; p++) {
      const result = this._allocateOnPage(p, w, h);
      if (result) return { page: p, ...result };
    }
    // All pages full — create a new one
    const newPage = this._createPage();
    const p = this.pages.length - 1;
    return { page: p, ...this._allocateOnPage(p, w, h)! };
  }

  private _allocateOnPage(pageIdx: number, w: number, h: number): { x: number; y: number } | null {
    const page = this.pages[pageIdx];

    // Try to fit on an existing shelf
    for (const shelf of page.shelves) {
      if (h <= shelf.height && shelf.cursorX + w <= ATLAS_SIZE) {
        const x = shelf.cursorX;
        shelf.cursorX += w;
        return { x, y: shelf.y };
      }
    }

    // Create a new shelf if there's vertical room
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
    return this.uvMap.has(imageId);
  }

  loadThumbnail(imageId: string, url: string): void {
    if (this.uvMap.has(imageId) || this.loadingSet.has(imageId)) return;
    this.loadingSet.add(imageId);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      this.loadingSet.delete(imageId);

      // Scale to fit MAX_THUMB_H, preserving aspect ratio
      const aspect = img.naturalWidth / img.naturalHeight;
      const drawH = Math.min(img.naturalHeight, MAX_THUMB_H);
      const drawW = Math.round(drawH * aspect);

      const slot = this._allocate(drawW, drawH);
      if (!slot) return;

      // Draw at natural aspect ratio into a temporary canvas
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

      // UV covers exactly the image pixels — no padding, no stretch
      const uv: UVRect = {
        u0: slot.x / ATLAS_SIZE,
        v0: (slot.y + drawH) / ATLAS_SIZE,
        u1: (slot.x + drawW) / ATLAS_SIZE,
        v1: slot.y / ATLAS_SIZE,
      };
      this.uvMap.set(imageId, { page: slot.page, uv });
    };
    img.onerror = () => {
      this.loadingSet.delete(imageId);
    };
    img.src = url;
  }

  pageCount(): number {
    return this.pages.length;
  }

  getPage(imageId: string): number {
    return this.uvMap.get(imageId)?.page ?? 0;
  }

  bindPage(page: number, textureUnit: number): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.pages[page].texture);
  }
}
