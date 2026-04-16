/**
 * WebGL torus viewport renderer.
 *
 * Two-tier resolution:
 *  - Zoomed out: 96px thumbnails packed in a texture atlas (batched draw calls)
 *  - Zoomed in: 1024px previews as individual textures (per-image draw calls)
 *
 * Transition threshold: when an image's rendered height exceeds PREVIEW_THRESHOLD
 * CSS pixels, the preview is loaded. The thumbnail stays visible until the
 * preview arrives, then the preview draws on top.
 *
 * Performance for 70k+ images:
 *  - Binary search on sorted strips/images for visibility
 *  - Thumbnails loaded on demand, concurrency-capped
 *  - Previews: LRU cache, max 50, evicted when scrolled away
 */

import type { StripLayout, PointOfView, Strip } from "../types";
import { TextureAtlas } from "./texture-atlas";
import { PreviewCache } from "./preview-cache";

// When an image renders taller than this in CSS pixels, load its preview
const PREVIEW_THRESHOLD = 150;

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
}

export class TorusViewport {
  private gl: WebGLRenderingContext;
  private canvas: HTMLCanvasElement;
  private program: WebGLProgram;
  private atlas: TextureAtlas;
  private previews: PreviewCache;

  // Buffers
  private quadBuf: WebGLBuffer;
  private offsetBuf: WebGLBuffer;
  private sizeBuf: WebGLBuffer;
  private uvBuf: WebGLBuffer;

  // Attribute locations
  private aPosition: number;
  private aOffset: number;
  private aSize: number;
  private aUV: number;

  // Uniform locations
  private uCamera: WebGLUniformLocation;
  private uZoom: WebGLUniformLocation;
  private uViewport: WebGLUniformLocation;
  private uAtlas: WebGLUniformLocation;

  // State
  private layout: StripLayout | null = null;
  private thumbnailBaseUrl = "";
  private animFrameId = 0;
  private stripYs: Float64Array = new Float64Array(0);

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

  setLayout(layout: StripLayout): void {
    this.layout = layout;
    const ys = new Float64Array(layout.strips.length);
    for (let i = 0; i < layout.strips.length; i++) {
      ys[i] = layout.strips[i].y;
    }
    this.stripYs = ys;
  }

  // ── Visibility: binary search on sorted strips/images ──

  private _firstVisibleStrip(viewTop: number): number {
    const strips = this.layout!.strips;
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

  /**
   * Collect visible images, split into atlas quads and preview quads.
   * An image qualifies for preview when its rendered height exceeds
   * PREVIEW_THRESHOLD CSS pixels.
   */
  private _collectVisible(pov: PointOfView): {
    atlasPages: Map<number, { offsets: number[]; sizes: number[]; uvs: number[]; count: number }>;
    previewQuads: PreviewQuad[];
  } {
    const atlasPages = new Map<number, { offsets: number[]; sizes: number[]; uvs: number[]; count: number }>();
    const previewQuads: PreviewQuad[] = [];

    if (!this.layout) return { atlasPages, previewQuads };

    const { strips, torus_width, torus_height } = this.layout;
    if (strips.length === 0 || torus_width === 0 || torus_height === 0) return { atlasPages, previewQuads };

    const aspect = this.canvas.width / this.canvas.height;
    const visW = pov.z;
    const visH = pov.z / aspect;

    // How many CSS pixels does one world unit occupy?
    const cssPixelsPerUnit = (this.canvas.clientHeight) / visH;

    // Draw bounds: exact viewport
    const viewLeft = pov.x - visW * 0.5;
    const viewRight = pov.x + visW * 0.5;
    const viewTop = pov.y - visH * 0.5;
    const viewBottom = pov.y + visH * 0.5;

    // Prefetch margin: 50% of viewport in each direction — pre-load
    // thumbnails before they scroll into view. Scales with zoom.
    const marginX = visW * 0.5;
    const marginY = visH * 0.5;
    const fetchLeft = viewLeft - marginX;
    const fetchRight = viewRight + marginX;
    const fetchTop = viewTop - marginY;
    const fetchBottom = viewBottom + marginY;

    const baseUrl = this.thumbnailBaseUrl;

    // Scan the fetch region (viewport + margin). Images in the margin
    // get their thumbnails loaded but aren't drawn.
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

        // Is this strip within the draw region (not just fetch)?
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

            // Prefetch thumbnail for everything in fetch region
            if (!this.atlas.isLoaded(img.id) && baseUrl) {
              this.atlas.loadThumbnail(img.id, `${baseUrl}/thumbnail/${img.id}`);
            }

            // Only draw if within the actual viewport
            const inView = stripVisible && wx + img.width >= viewLeft && wx <= viewRight;
            if (!inView) continue;

            if (needsPreview) {
              if (!this.previews.isRequested(img.id) && baseUrl) {
                this.previews.load(img.id, `${baseUrl}/preview/${img.id}`);
              }
              if (this.previews.has(img.id)) {
                previewQuads.push({ id: img.id, wx, wy, width: img.width, height: sh });
                continue;
              }
            }

            const entry = this.atlas.getUV(img.id);
            const { uv } = entry;
            const page = entry.page;

            let bucket = atlasPages.get(page);
            if (!bucket) {
              bucket = { offsets: [], sizes: [], uvs: [], count: 0 };
              atlasPages.set(page, bucket);
            }

            for (let v = 0; v < 6; v++) {
              bucket.offsets.push(wx, wy);
              bucket.sizes.push(img.width, sh);
              bucket.uvs.push(uv.u0, uv.v0, uv.u1, uv.v1);
            }
            bucket.count += 6;
          }
        }
      }
    }

    return { atlasPages, previewQuads };
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

    if (!this.layout) return;

    const { atlasPages, previewQuads } = this._collectVisible(pov);

    gl.useProgram(this.program);
    gl.uniform2f(this.uCamera, pov.x, pov.y);
    gl.uniform1f(this.uZoom, pov.z);
    gl.uniform2f(this.uViewport, canvas.width, canvas.height);

    // Pass 1: Atlas thumbnails (batched per page)
    for (const [page, bucket] of atlasPages) {
      if (bucket.count === 0) continue;

      this.atlas.bindPage(page, 0);
      gl.uniform1i(this.uAtlas, 0);

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

    // Pass 2: Preview textures (individual draw calls, drawn on top)
    for (const pq of previewQuads) {
      const entry = this.previews.get(pq.id);
      if (!entry) continue;

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, entry.texture);
      gl.uniform1i(this.uAtlas, 0);

      // Full-texture UVs: [0,1] x [0,1]
      const posData = new Float32Array(QUAD);
      const offData = new Float32Array(12);
      const szData = new Float32Array(12);
      const uvData = new Float32Array(24);

      for (let v = 0; v < 6; v++) {
        offData[v * 2] = pq.wx;
        offData[v * 2 + 1] = pq.wy;
        szData[v * 2] = pq.width;
        szData[v * 2 + 1] = pq.height;
        uvData[v * 4] = 0;     // u0
        uvData[v * 4 + 1] = 1; // v0 (bottom)
        uvData[v * 4 + 2] = 1; // u1
        uvData[v * 4 + 3] = 0; // v1 (top)
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, posData, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aPosition);
      gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.offsetBuf);
      gl.bufferData(gl.ARRAY_BUFFER, offData, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aOffset);
      gl.vertexAttribPointer(this.aOffset, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuf);
      gl.bufferData(gl.ARRAY_BUFFER, szData, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aSize);
      gl.vertexAttribPointer(this.aSize, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, uvData, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aUV);
      gl.vertexAttribPointer(this.aUV, 4, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
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
