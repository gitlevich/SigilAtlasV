/**
 * WebGL torus viewport renderer.
 *
 * Renders a flat viewport into a torus rectangle with seamless wraparound.
 * Images are drawn as textured quads from a texture atlas.
 *
 * Camera (PointOfView): x,y pan with modulo wrap, z controls zoom level.
 * At z near 0: one image fills the frame. At high z: entire torus visible.
 *
 * Performance design for 70k+ images:
 *  - Strips sorted by y, images within strip sorted by x → binary search
 *  - Single visibility pass per frame, partitioned by atlas page
 *  - Thumbnails loaded only for visible images, with concurrency cap
 *  - Pre-allocated typed arrays avoid GC pressure
 */

import type { StripLayout, PointOfView, Strip } from "../types";
import { TextureAtlas } from "./texture-atlas";

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

// Unit quad vertices: two triangles
const QUAD = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1];

export class TorusViewport {
  private gl: WebGLRenderingContext;
  private canvas: HTMLCanvasElement;
  private program: WebGLProgram;
  private atlas: TextureAtlas;

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

  // Strip y-index for binary search (built on setLayout)
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

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
  }

  setThumbnailBaseUrl(url: string): void {
    this.thumbnailBaseUrl = url;
  }

  setLayout(layout: StripLayout): void {
    this.layout = layout;
    // Build y-index for binary search — strips are sorted by y ascending
    const ys = new Float64Array(layout.strips.length);
    for (let i = 0; i < layout.strips.length; i++) {
      ys[i] = layout.strips[i].y;
    }
    this.stripYs = ys;
    // Thumbnails are NOT loaded here. They load on demand during render.
  }

  // ── Visibility: binary search on sorted strips/images ──

  /** First strip index whose y + height > viewTop. */
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

  /** First image index in strip whose x + width > viewLeft. */
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
   * Collect all visible quads in one pass. Returns per-page arrays.
   * Also queues thumbnail loads for newly visible images.
   */
  private _collectVisible(pov: PointOfView): Map<number, { offsets: number[]; sizes: number[]; uvs: number[]; count: number }> {
    const pages = new Map<number, { offsets: number[]; sizes: number[]; uvs: number[]; count: number }>();

    if (!this.layout) return pages;

    const { strips, torus_width, torus_height } = this.layout;
    if (strips.length === 0 || torus_width === 0 || torus_height === 0) return pages;

    const aspect = this.canvas.width / this.canvas.height;
    const visW = pov.z;
    const visH = pov.z / aspect;

    const viewLeft = pov.x - visW * 0.5;
    const viewRight = pov.x + visW * 0.5;
    const viewTop = pov.y - visH * 0.5;
    const viewBottom = pov.y + visH * 0.5;

    const baseUrl = this.thumbnailBaseUrl;

    // Iterate torus-wrap copies that overlap the view
    for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
      const wrapTop = viewTop - dy;
      const wrapBottom = viewBottom - dy;

      // Binary search: first strip visible at this y-wrap
      const si0 = this._firstVisibleStrip(wrapTop);

      for (let si = si0; si < strips.length; si++) {
        const strip = strips[si];
        const wy = strip.y + dy;
        if (wy > viewBottom) break; // past view — done with this wrap
        if (wy + strip.height < viewTop) continue;

        const sh = strip.height;

        for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
          const shiftedLeft = viewLeft - dx;
          const shiftedRight = viewRight - dx;

          // Binary search: first image visible at this x-wrap
          const ii0 = this._firstVisibleImage(strip, shiftedLeft);

          for (let ii = ii0; ii < strip.images.length; ii++) {
            const img = strip.images[ii];
            if (img.x > shiftedRight) break; // past view — done with this x-wrap
            if (img.x + img.width < shiftedLeft) continue;

            const wx = img.x + dx;

            // Queue thumbnail load if needed
            if (!this.atlas.isLoaded(img.id) && baseUrl) {
              this.atlas.loadThumbnail(img.id, `${baseUrl}/thumbnail/${img.id}`);
            }

            const entry = this.atlas.getUV(img.id);
            const { uv } = entry;
            const page = entry.page;

            let bucket = pages.get(page);
            if (!bucket) {
              bucket = { offsets: [], sizes: [], uvs: [], count: 0 };
              pages.set(page, bucket);
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

    return pages;
  }

  render(pov: PointOfView): void {
    const gl = this.gl;
    const { canvas } = this;

    // Resize canvas to match display
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

    // Single pass: collect visible quads partitioned by atlas page
    const pages = this._collectVisible(pov);
    if (pages.size === 0) return;

    gl.useProgram(this.program);
    gl.uniform2f(this.uCamera, pov.x, pov.y);
    gl.uniform1f(this.uZoom, pov.z);
    gl.uniform2f(this.uViewport, canvas.width, canvas.height);

    for (const [page, bucket] of pages) {
      if (bucket.count === 0) continue;

      this.atlas.bindPage(page, 0);
      gl.uniform1i(this.uAtlas, 0);

      // Position attribute: repeat unit quad per vertex
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
