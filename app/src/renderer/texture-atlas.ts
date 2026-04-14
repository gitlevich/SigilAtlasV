/**
 * Texture atlas for thumbnail images.
 *
 * Packs thumbnails into GPU textures and provides UV coordinates for each image.
 * Images are resized to fit atlas slots before upload.
 */

const ATLAS_SIZE = 4096;
const THUMB_H = 96;         // Height of each thumbnail in atlas (= strip_height analog)
const SLOT_W = 192;         // Slot width: accommodates up to 2:1 panoramas at THUMB_H
const SLOT_H = THUMB_H;
const SLOTS_PER_ROW = Math.floor(ATLAS_SIZE / SLOT_W);
const ROWS_PER_PAGE = Math.floor(ATLAS_SIZE / SLOT_H);
const MAX_SLOTS = SLOTS_PER_ROW * ROWS_PER_PAGE;

export interface UVRect {
  u0: number;
  v0: number;
  u1: number;
  v1: number;
}

interface AtlasPage {
  texture: WebGLTexture;
  nextSlot: number;
}

export class TextureAtlas {
  private gl: WebGLRenderingContext;
  private pages: AtlasPage[] = [];
  private uvMap: Map<string, { page: number; uv: UVRect }> = new Map();
  private loadingSet: Set<string> = new Set();
  private resizeCanvas: HTMLCanvasElement;
  private resizeCtx: CanvasRenderingContext2D;

  placeholderUV: UVRect = { u0: 0, v0: 0, u1: 0, v1: 0 };
  placeholderPage = 0;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this.resizeCanvas = document.createElement("canvas");
    this.resizeCanvas.width = SLOT_W;
    this.resizeCanvas.height = SLOT_H;
    this.resizeCtx = this.resizeCanvas.getContext("2d")!;

    this._createPage();
    this._initPlaceholder();
  }

  private _createPage(): AtlasPage {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      ATLAS_SIZE, ATLAS_SIZE, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const page: AtlasPage = { texture: tex, nextSlot: 0 };
    this.pages.push(page);
    return page;
  }

  private _initPlaceholder(): void {
    const gl = this.gl;
    const pixel = new Uint8Array([80, 80, 90, 255]);
    gl.bindTexture(gl.TEXTURE_2D, this.pages[0].texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    this.pages[0].nextSlot = 1;
    this.placeholderUV = { u0: 0, v0: 0, u1: 1 / ATLAS_SIZE, v1: 1 / ATLAS_SIZE };
  }

  private _allocateSlot(): { page: number; x: number; y: number } | null {
    for (let p = 0; p < this.pages.length; p++) {
      if (this.pages[p].nextSlot < MAX_SLOTS) {
        const slot = this.pages[p].nextSlot++;
        const col = slot % SLOTS_PER_ROW;
        const row = Math.floor(slot / SLOTS_PER_ROW);
        return { page: p, x: col * SLOT_W, y: row * SLOT_H };
      }
    }
    const page = this._createPage();
    const p = this.pages.length - 1;
    page.nextSlot = 1;
    return { page: p, x: 0, y: 0 };
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
      const slot = this._allocateSlot();
      if (!slot) return;

      // Constrain height to SLOT_H, maintain aspect ratio (!undistorted).
      // Image fills its atlas region completely — no padding (!gapless).
      const srcAspect = img.naturalWidth / img.naturalHeight;
      const drawH = SLOT_H;
      const drawW = Math.min(Math.round(SLOT_H * srcAspect), SLOT_W);

      const ctx = this.resizeCtx;
      this.resizeCanvas.width = drawW;
      this.resizeCanvas.height = drawH;
      ctx.drawImage(img, 0, 0, drawW, drawH);

      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.pages[slot.page].texture);
      gl.texSubImage2D(
        gl.TEXTURE_2D, 0,
        slot.x, slot.y,
        gl.RGBA, gl.UNSIGNED_BYTE,
        this.resizeCanvas,
      );

      // Reset canvas
      this.resizeCanvas.width = SLOT_W;
      this.resizeCanvas.height = SLOT_H;

      // UV covers exactly what was drawn — correct aspect, no gaps
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

  bindPage(page: number, textureUnit: number): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, this.pages[page].texture);
  }
}
