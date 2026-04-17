/**
 * WebGL torus viewport renderer.
 *
 * Two layout modes:
 *  - Strip mode (@TimeLike): binary-search packed strips, native aspect ratios.
 *  - Grid mode (@SpaceLike): uniform square cells, per-image gravity tweening.
 *
 * Two-tier resolution in both modes:
 *  - Zoomed out: 96px thumbnails packed in a texture atlas (batched draw calls).
 *  - Zoomed in: 1024px previews as individual textures (per-image draw calls).
 *
 * In @SpaceLike, every image is rendered as a center-cropped square — the UV
 * rect is narrowed on the longer axis so the shader samples only the central
 * square of the thumbnail. No image re-encoding.
 */

import type {
  AnyLayout,
  StripLayout,
  SpaceLikeLayout,
  PointOfView,
  Strip,
} from "../types";
import { isSpaceLikeLayout } from "../types";
import { TextureAtlas, type UVRect } from "./texture-atlas";
import { PreviewCache } from "./preview-cache";

// When an image renders taller than this in CSS pixels, load its preview
const PREVIEW_THRESHOLD = 150;

// Tween duration for @SpaceLike gravity-field settling
const TWEEN_MS = 500;

const VERT_SRC = `
attribute vec2 a_position;
attribute vec2 a_offset;
attribute vec2 a_size;
attribute vec4 a_uv;

uniform vec2 u_camera;
uniform float u_zoom;
uniform vec2 u_viewport;

varying vec2 v_texcoord;

void main() {
  vec2 corner = a_offset + a_position * a_size;
  vec2 rel = corner - u_camera;
  float aspect = u_viewport.x / u_viewport.y;
  float visibleHeight = u_zoom / aspect;
  vec2 ndc;
  ndc.x = rel.x / (u_zoom * 0.5);
  ndc.y = rel.y / (visibleHeight * 0.5);
  gl_Position = vec4(ndc, 0.0, 1.0);
  v_texcoord = mix(a_uv.xy, a_uv.zw, a_position);
}
`;

const FRAG_SRC = `
precision mediump float;
varying vec2 v_texcoord;
uniform sampler2D u_atlas;
void main() {
  gl_FragColor = texture2D(u_atlas, v_texcoord);
}
`;

const QUAD = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1];

interface PreviewQuad {
  id: string;
  wx: number;
  wy: number;
  width: number;
  height: number;
  uvU0: number;
  uvV0: number;
  uvU1: number;
  uvV1: number;
}

interface SpriteTween {
  startCol: number;
  startRow: number;
  targetCol: number;
  targetRow: number;
  tweenStart: number;
}

export class TorusViewport {
  private gl: WebGLRenderingContext;
  private canvas: HTMLCanvasElement;
  private program: WebGLProgram;
  private atlas: TextureAtlas;
  private previews: PreviewCache;

  private quadBuf: WebGLBuffer;
  private offsetBuf: WebGLBuffer;
  private sizeBuf: WebGLBuffer;
  private uvBuf: WebGLBuffer;

  private aPosition: number;
  private aOffset: number;
  private aSize: number;
  private aUV: number;

  private uCamera: WebGLUniformLocation;
  private uZoom: WebGLUniformLocation;
  private uViewport: WebGLUniformLocation;
  private uAtlas: WebGLUniformLocation;

  // Layout state: at most one is non-null at a time
  private stripLayout: StripLayout | null = null;
  private gridLayout: SpaceLikeLayout | null = null;

  // @SpaceLike per-image tween state — persists across layout changes
  private tweens: Map<string, SpriteTween> = new Map();

  // Strip mode: sorted y-coords for binary search
  private stripYs: Float64Array = new Float64Array(0);

  private thumbnailBaseUrl = "";
  private animFrameId = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: false, alpha: false })!;
    if (!gl) throw new Error("WebGL not supported");
    this.gl = gl;

    this.program = this._createProgram(VERT_SRC, FRAG_SRC);
    gl.useProgram(this.program);

    this.aPosition = gl.getAttribLocation(this.program, "a_position");
    this.aOffset = gl.getAttribLocation(this.program, "a_offset");
    this.aSize = gl.getAttribLocation(this.program, "a_size");
    this.aUV = gl.getAttribLocation(this.program, "a_uv");

    this.uCamera = gl.getUniformLocation(this.program, "u_camera")!;
    this.uZoom = gl.getUniformLocation(this.program, "u_zoom")!;
    this.uViewport = gl.getUniformLocation(this.program, "u_viewport")!;
    this.uAtlas = gl.getUniformLocation(this.program, "u_atlas")!;

    const quad = new Float32Array(QUAD);
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    this.offsetBuf = gl.createBuffer()!;
    this.sizeBuf = gl.createBuffer()!;
    this.uvBuf = gl.createBuffer()!;

    this.atlas = new TextureAtlas(gl);
    this.previews = new PreviewCache(gl);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
  }

  setThumbnailBaseUrl(url: string): void {
    this.thumbnailBaseUrl = url;
  }

  setLayout(layout: AnyLayout): void {
    if (isSpaceLikeLayout(layout)) {
      this._setGridLayout(layout);
    } else {
      this._setStripLayout(layout);
    }
  }

  private _setStripLayout(layout: StripLayout): void {
    this.gridLayout = null;
    this.tweens.clear();
    this.stripLayout = layout;
    const ys = new Float64Array(layout.strips.length);
    for (let i = 0; i < layout.strips.length; i++) ys[i] = layout.strips[i].y;
    this.stripYs = ys;
  }

  private _setGridLayout(layout: SpaceLikeLayout): void {
    this.stripLayout = null;
    this.gridLayout = layout;
    const now = performance.now();

    // First occurrence of each id becomes the "primary" animated sprite.
    // Duplicates (pad cells) draw statically at their cell with no tween.
    const primaries = new Map<string, { col: number; row: number }>();
    for (const p of layout.positions) {
      if (!primaries.has(p.id)) primaries.set(p.id, { col: p.col, row: p.row });
    }

    const cols = layout.cols;
    const rows = layout.rows;

    const next: Map<string, SpriteTween> = new Map();
    for (const [id, { col, row }] of primaries) {
      const prev = this.tweens.get(id);
      let sCol: number;
      let sRow: number;
      if (prev) {
        // Continue from current interpolated position
        const t = Math.min(1, (now - prev.tweenStart) / TWEEN_MS);
        const eased = 0.5 - 0.5 * Math.cos(t * Math.PI);
        sCol = prev.startCol + (prev.targetCol - prev.startCol) * eased;
        sRow = prev.startRow + (prev.targetRow - prev.startRow) * eased;
      } else {
        // New sprite — appear at target instantly (no tween)
        sCol = col;
        sRow = row;
      }

      // Choose the short path across torus wrap
      let dCol = col - sCol;
      if (dCol > cols * 0.5) dCol -= cols;
      else if (dCol < -cols * 0.5) dCol += cols;
      let dRow = row - sRow;
      if (dRow > rows * 0.5) dRow -= rows;
      else if (dRow < -rows * 0.5) dRow += rows;

      next.set(id, {
        startCol: sCol,
        startRow: sRow,
        targetCol: sCol + dCol,
        targetRow: sRow + dRow,
        tweenStart: now,
      });
    }

    this.tweens = next;
  }

  // ── Strip mode (TimeLike) ──

  private _firstVisibleStrip(viewTop: number): number {
    const strips = this.stripLayout!.strips;
    let lo = 0, hi = strips.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (strips[mid].y + strips[mid].height <= viewTop) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  private _firstVisibleImage(strip: Strip, viewLeft: number): number {
    const imgs = strip.images;
    let lo = 0, hi = imgs.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (imgs[mid].x + imgs[mid].width <= viewLeft) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  private _collectStripVisible(pov: PointOfView): {
    atlasPages: Map<number, VertexBucket>;
    previewQuads: PreviewQuad[];
  } {
    const atlasPages = new Map<number, VertexBucket>();
    const previewQuads: PreviewQuad[] = [];

    const layout = this.stripLayout;
    if (!layout) return { atlasPages, previewQuads };
    const { strips, torus_width, torus_height } = layout;
    if (strips.length === 0 || torus_width === 0 || torus_height === 0) return { atlasPages, previewQuads };

    const aspect = this.canvas.width / this.canvas.height;
    const visW = pov.z;
    const visH = pov.z / aspect;
    const cssPixelsPerUnit = this.canvas.clientHeight / visH;

    const viewLeft = pov.x - visW * 0.5;
    const viewRight = pov.x + visW * 0.5;
    const viewTop = pov.y - visH * 0.5;
    const viewBottom = pov.y + visH * 0.5;

    const marginX = visW * 0.5;
    const marginY = visH * 0.5;
    const fetchLeft = viewLeft - marginX;
    const fetchRight = viewRight + marginX;
    const fetchTop = viewTop - marginY;
    const fetchBottom = viewBottom + marginY;

    const baseUrl = this.thumbnailBaseUrl;

    for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
      const si0 = this._firstVisibleStrip(fetchTop - dy);

      for (let si = si0; si < strips.length; si++) {
        const strip = strips[si];
        const wy = strip.y + dy;
        if (wy > fetchBottom) break;
        if (wy + strip.height < fetchTop) continue;

        const sh = strip.height;
        const renderedH = sh * cssPixelsPerUnit;
        const needsPreview = renderedH > PREVIEW_THRESHOLD;
        const stripVisible = wy + sh >= viewTop && wy <= viewBottom;

        for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
          const fetchShiftedLeft = fetchLeft - dx;
          const fetchShiftedRight = fetchRight - dx;
          const ii0 = this._firstVisibleImage(strip, fetchShiftedLeft);

          for (let ii = ii0; ii < strip.images.length; ii++) {
            const img = strip.images[ii];
            if (img.x > fetchShiftedRight) break;
            if (img.x + img.width < fetchShiftedLeft) continue;

            const wx = img.x + dx;

            if (!this.atlas.isLoaded(img.id) && baseUrl) {
              this.atlas.loadThumbnail(img.id, `${baseUrl}/thumbnail/${img.id}`);
            }

            const inView = stripVisible && wx + img.width >= viewLeft && wx <= viewRight;
            if (!inView) continue;

            if (needsPreview) {
              if (!this.previews.isRequested(img.id) && baseUrl) {
                this.previews.load(img.id, `${baseUrl}/preview/${img.id}`);
              }
              if (this.previews.has(img.id)) {
                previewQuads.push({
                  id: img.id, wx, wy, width: img.width, height: sh,
                  uvU0: 0, uvV0: 1, uvU1: 1, uvV1: 0,
                });
                continue;
              }
            }

            const entry = this.atlas.getUV(img.id);
            this._pushQuad(atlasPages, entry.page, wx, wy, img.width, sh,
              entry.uv.u0, entry.uv.v0, entry.uv.u1, entry.uv.v1);
          }
        }
      }
    }

    return { atlasPages, previewQuads };
  }

  // ── Grid mode (SpaceLike) ──

  private _squareCropUV(full: UVRect): UVRect {
    // full spans the image at its native aspect. Atlas convention: u1 > u0,
    // v0 > v1 (flipped). Shrink the longer axis symmetrically around center
    // to produce a square-cropped UV, so the sprite samples the center square.
    const du = full.u1 - full.u0;
    const dvAbs = Math.abs(full.v0 - full.v1);
    if (du <= 0 || dvAbs <= 0) return full;

    if (du > dvAbs) {
      const uMid = (full.u0 + full.u1) * 0.5;
      const half = dvAbs * 0.5;
      return { u0: uMid - half, v0: full.v0, u1: uMid + half, v1: full.v1 };
    } else if (dvAbs > du) {
      const vMid = (full.v0 + full.v1) * 0.5;
      const halfSigned = du * 0.5 * (full.v0 >= full.v1 ? 1 : -1);
      return { u0: full.u0, v0: vMid + halfSigned, u1: full.u1, v1: vMid - halfSigned };
    }
    return full;
  }

  private _spriteCurrentPos(id: string, now: number): { col: number; row: number } {
    const tw = this.tweens.get(id);
    if (!tw) return { col: 0, row: 0 };
    const t = Math.min(1, (now - tw.tweenStart) / TWEEN_MS);
    const eased = 0.5 - 0.5 * Math.cos(t * Math.PI);
    return {
      col: tw.startCol + (tw.targetCol - tw.startCol) * eased,
      row: tw.startRow + (tw.targetRow - tw.startRow) * eased,
    };
  }

  private _collectGridVisible(pov: PointOfView): {
    atlasPages: Map<number, VertexBucket>;
    previewQuads: PreviewQuad[];
  } {
    const atlasPages = new Map<number, VertexBucket>();
    const previewQuads: PreviewQuad[] = [];

    const layout = this.gridLayout;
    if (!layout) return { atlasPages, previewQuads };
    const { positions, cell_size, cols, rows, torus_width, torus_height } = layout;
    if (positions.length === 0) return { atlasPages, previewQuads };

    const aspect = this.canvas.width / this.canvas.height;
    const visW = pov.z;
    const visH = pov.z / aspect;
    const cssPixelsPerUnit = this.canvas.clientHeight / visH;
    const renderedH = cell_size * cssPixelsPerUnit;
    const needsPreview = renderedH > PREVIEW_THRESHOLD;

    const viewLeft = pov.x - visW * 0.5;
    const viewRight = pov.x + visW * 0.5;
    const viewTop = pov.y - visH * 0.5;
    const viewBottom = pov.y + visH * 0.5;

    const marginX = visW * 0.5;
    const marginY = visH * 0.5;
    const fetchLeft = viewLeft - marginX;
    const fetchRight = viewRight + marginX;
    const fetchTop = viewTop - marginY;
    const fetchBottom = viewBottom + marginY;

    const baseUrl = this.thumbnailBaseUrl;
    const now = performance.now();

    // Track which ids have already been rendered as the tweened primary —
    // later duplicates draw statically at their target cell.
    const seenIds = new Set<string>();

    for (const pos of positions) {
      const isPrimary = !seenIds.has(pos.id);
      seenIds.add(pos.id);

      let curCol: number;
      let curRow: number;
      if (isPrimary) {
        const cp = this._spriteCurrentPos(pos.id, now);
        curCol = cp.col;
        curRow = cp.row;
      } else {
        curCol = pos.col;
        curRow = pos.row;
      }

      // Wrap current position into [0, cols) / [0, rows)
      curCol = ((curCol % cols) + cols) % cols;
      curRow = ((curRow % rows) + rows) % rows;

      const baseWx = curCol * cell_size;
      const baseWy = curRow * cell_size;

      // Render tiled across torus wraps
      let requested = false;
      for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
        const wy = baseWy + dy;
        if (wy + cell_size < fetchTop || wy > fetchBottom) continue;
        for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
          const wx = baseWx + dx;
          if (wx + cell_size < fetchLeft || wx > fetchRight) continue;

          if (!requested && !this.atlas.isLoaded(pos.id) && baseUrl) {
            this.atlas.loadThumbnail(pos.id, `${baseUrl}/thumbnail/${pos.id}`);
            requested = true;
          }

          const inView = wx + cell_size >= viewLeft && wx <= viewRight
            && wy + cell_size >= viewTop && wy <= viewBottom;
          if (!inView) continue;

          if (needsPreview) {
            if (!this.previews.isRequested(pos.id) && baseUrl) {
              this.previews.load(pos.id, `${baseUrl}/preview/${pos.id}`);
            }
            if (this.previews.has(pos.id)) {
              // Square-crop preview: full texture UV [0,1] x [1,0], crop to center
              const cropped = this._squareCropUV({ u0: 0, v0: 1, u1: 1, v1: 0 });
              previewQuads.push({
                id: pos.id, wx, wy, width: cell_size, height: cell_size,
                uvU0: cropped.u0, uvV0: cropped.v0,
                uvU1: cropped.u1, uvV1: cropped.v1,
              });
              continue;
            }
          }

          const entry = this.atlas.getUV(pos.id);
          const cropped = this._squareCropUV(entry.uv);
          this._pushQuad(atlasPages, entry.page, wx, wy, cell_size, cell_size,
            cropped.u0, cropped.v0, cropped.u1, cropped.v1);
        }
      }
    }

    return { atlasPages, previewQuads };
  }

  private _pushQuad(
    atlasPages: Map<number, VertexBucket>, page: number,
    wx: number, wy: number, w: number, h: number,
    u0: number, v0: number, u1: number, v1: number,
  ): void {
    let bucket = atlasPages.get(page);
    if (!bucket) {
      bucket = { offsets: [], sizes: [], uvs: [], count: 0 };
      atlasPages.set(page, bucket);
    }
    for (let v = 0; v < 6; v++) {
      bucket.offsets.push(wx, wy);
      bucket.sizes.push(w, h);
      bucket.uvs.push(u0, v0, u1, v1);
    }
    bucket.count += 6;
  }

  render(pov: PointOfView): void {
    const gl = this.gl;
    const { canvas } = this;

    const dpr = window.devicePixelRatio || 1;
    const displayW = canvas.clientWidth * dpr;
    const displayH = canvas.clientHeight * dpr;
    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width = displayW;
      canvas.height = displayH;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (!this.stripLayout && !this.gridLayout) return;

    const { atlasPages, previewQuads } = this.stripLayout
      ? this._collectStripVisible(pov)
      : this._collectGridVisible(pov);

    gl.useProgram(this.program);
    gl.uniform2f(this.uCamera, pov.x, pov.y);
    gl.uniform1f(this.uZoom, pov.z);
    gl.uniform2f(this.uViewport, canvas.width, canvas.height);

    // Pass 1: atlas thumbnails (batched per page)
    for (const [page, bucket] of atlasPages) {
      if (bucket.count === 0) continue;
      this.atlas.bindPage(page, 0);
      gl.uniform1i(this.uAtlas, 0);
      this._uploadAndDraw(bucket);
    }

    // Pass 2: previews (individual draw calls, on top)
    for (const pq of previewQuads) {
      const entry = this.previews.get(pq.id);
      if (!entry) continue;

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, entry.texture);
      gl.uniform1i(this.uAtlas, 0);

      const bucket: VertexBucket = { offsets: [], sizes: [], uvs: [], count: 0 };
      for (let v = 0; v < 6; v++) {
        bucket.offsets.push(pq.wx, pq.wy);
        bucket.sizes.push(pq.width, pq.height);
        bucket.uvs.push(pq.uvU0, pq.uvV0, pq.uvU1, pq.uvV1);
      }
      bucket.count = 6;
      this._uploadAndDraw(bucket);
    }
  }

  private _uploadAndDraw(bucket: VertexBucket): void {
    const gl = this.gl;
    const posData = new Float32Array(bucket.count * 2);
    for (let i = 0; i < bucket.count; i++) {
      const vi = i % 6;
      posData[i * 2] = QUAD[vi * 2];
      posData[i * 2 + 1] = QUAD[vi * 2 + 1];
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, posData, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.offsetBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bucket.offsets), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aOffset);
    gl.vertexAttribPointer(this.aOffset, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bucket.sizes), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aSize);
    gl.vertexAttribPointer(this.aSize, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bucket.uvs), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aUV);
    gl.vertexAttribPointer(this.aUV, 4, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, bucket.count);
  }

  startRenderLoop(getPov: () => PointOfView): void {
    const loop = () => {
      this.render(getPov());
      this.animFrameId = requestAnimationFrame(loop);
    };
    this.animFrameId = requestAnimationFrame(loop);
  }

  stopRenderLoop(): void {
    cancelAnimationFrame(this.animFrameId);
  }

  private _createProgram(vsrc: string, fsrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = this._compileShader(gl.VERTEX_SHADER, vsrc);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("Program link error: " + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error("Shader compile error: " + gl.getShaderInfoLog(s));
    }
    return s;
  }
}

interface VertexBucket {
  offsets: number[];
  sizes: number[];
  uvs: number[];
  count: number;
}
