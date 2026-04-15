/**
 * WebGL torus viewport renderer.
 *
 * Renders a flat viewport into a torus rectangle with seamless wraparound.
 * Images are drawn as instanced textured quads from a texture atlas.
 *
 * Camera (PointOfView): x,y pan with modulo wrap, z controls zoom level.
 * At z near 0: one image fills the frame. At high z: entire torus visible.
 */

import type { StripLayout, PointOfView, ImagePosition, Strip } from "../types";
import { TextureAtlas } from "./texture-atlas";

// Inline shaders (vite will handle this in production via ?raw imports)
const VERT_SRC = `
attribute vec2 a_position;
attribute vec2 a_offset;
attribute vec2 a_size;
attribute vec4 a_uv;

uniform vec2 u_camera;
uniform float u_zoom;
uniform vec2 u_viewport;
uniform vec2 u_torus;

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

interface InstanceData {
  offsets: Float32Array;
  sizes: Float32Array;
  uvs: Float32Array;
  count: number;
}

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
  private uTorus: WebGLUniformLocation;
  private uAtlas: WebGLUniformLocation;

  // State
  private layout: StripLayout | null = null;
  private thumbnailBaseUrl = "";
  private animFrameId = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: false, alpha: false })!;
    if (!gl) throw new Error("WebGL not supported");
    this.gl = gl;

    // Compile shaders
    this.program = this._createProgram(VERT_SRC, FRAG_SRC);
    gl.useProgram(this.program);

    // Attribute locations
    this.aPosition = gl.getAttribLocation(this.program, "a_position");
    this.aOffset = gl.getAttribLocation(this.program, "a_offset");
    this.aSize = gl.getAttribLocation(this.program, "a_size");
    this.aUV = gl.getAttribLocation(this.program, "a_uv");

    // Uniform locations
    this.uCamera = gl.getUniformLocation(this.program, "u_camera")!;
    this.uZoom = gl.getUniformLocation(this.program, "u_zoom")!;
    this.uViewport = gl.getUniformLocation(this.program, "u_viewport")!;
    this.uTorus = gl.getUniformLocation(this.program, "u_torus")!;
    this.uAtlas = gl.getUniformLocation(this.program, "u_atlas")!;

    // Unit quad: two triangles
    const quad = new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]);
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    // Instance buffers (will be filled per frame)
    this.offsetBuf = gl.createBuffer()!;
    this.sizeBuf = gl.createBuffer()!;
    this.uvBuf = gl.createBuffer()!;

    // Texture atlas
    this.atlas = new TextureAtlas(gl);

    // GL state
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
  }

  setThumbnailBaseUrl(url: string): void {
    this.thumbnailBaseUrl = url;
  }

  setLayout(layout: StripLayout): void {
    this.layout = layout;
    // Trigger thumbnail loading for visible images
    this._loadThumbnails(layout);
  }

  private _loadThumbnails(layout: StripLayout): void {
    for (const strip of layout.strips) {
      for (const img of strip.images) {
        if (!this.atlas.isLoaded(img.id)) {
          const url = `${this.thumbnailBaseUrl}/thumbnail/${img.id}`;
          this.atlas.loadThumbnail(img.id, url);
        }
      }
    }
  }

  private _buildInstancesForPage(pov: PointOfView, targetPage: number): InstanceData {
    return this._buildInstancesInner(pov, targetPage);
  }

  private _buildInstances(pov: PointOfView): InstanceData {
    return this._buildInstancesInner(pov, null);
  }

  private _buildInstancesInner(pov: PointOfView, targetPage: number | null): InstanceData {
    if (!this.layout) return { offsets: new Float32Array(0), sizes: new Float32Array(0), uvs: new Float32Array(0), count: 0 };

    const { strips, torus_width, torus_height } = this.layout;
    const aspect = this.canvas.width / this.canvas.height;
    const visW = pov.z;
    const visH = pov.z / aspect;
    const margin = Math.max(visW, visH) * 0.5;

    // Collect visible images
    const offsets: number[] = [];
    const sizes: number[] = [];
    const uvs: number[] = [];

    // Draw all torus-wrapped copies that fall within the visible window.
    // This prevents overlaps and gaps at the seam.
    const viewLeft = pov.x - visW * 0.5 - margin;
    const viewRight = pov.x + visW * 0.5 + margin;
    const viewTop = pov.y - visH * 0.5 - margin;
    const viewBottom = pov.y + visH * 0.5 + margin;

    for (const strip of strips) {
      const sy = strip.y;
      const sh = strip.height;

      // Emit strip at all y-wraps that are visible
      for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
        const wy = sy + dy;
        if (wy + sh < viewTop || wy > viewBottom) continue;

        for (const img of strip.images) {
          // Emit image at all x-wraps that are visible
          for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
            const wx = img.x + dx;
            if (wx + img.width < viewLeft || wx > viewRight) continue;

            const entry = this.atlas.getUV(img.id);
            if (targetPage !== null && entry.page !== targetPage) continue;
            const { uv } = entry;

            for (let v = 0; v < 6; v++) {
              offsets.push(wx, wy);
              sizes.push(img.width, sh);
              uvs.push(uv.u0, uv.v0, uv.u1, uv.v1);
            }
          }
        }
      }
    }

    const count = offsets.length / 2 / 6; // number of quads
    return {
      offsets: new Float32Array(offsets),
      sizes: new Float32Array(sizes),
      uvs: new Float32Array(uvs),
      count: count * 6, // vertex count
    };
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

    const instances = this._buildInstances(pov);
    if (instances.count === 0) return;

    gl.useProgram(this.program);

    // Uniforms
    gl.uniform2f(this.uCamera, pov.x, pov.y);
    gl.uniform1f(this.uZoom, pov.z);
    gl.uniform2f(this.uViewport, canvas.width, canvas.height);
    gl.uniform2f(this.uTorus, this.layout.torus_width, this.layout.torus_height);

    // Render per atlas page — each page is a separate draw call
    const pageCount = this.atlas.pageCount();
    for (let page = 0; page < pageCount; page++) {
      const pageInstances = this._buildInstancesForPage(pov, page);
      if (pageInstances.count === 0) continue;

      this.atlas.bindPage(page, 0);
      gl.uniform1i(this.uAtlas, 0);

      // Position attribute
      const quadData = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1];
      const posData = new Float32Array(pageInstances.count * 2);
      for (let i = 0; i < pageInstances.count; i++) {
        const vi = i % 6;
        posData[i * 2] = quadData[vi * 2];
        posData[i * 2 + 1] = quadData[vi * 2 + 1];
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, posData, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aPosition);
      gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.offsetBuf);
      gl.bufferData(gl.ARRAY_BUFFER, pageInstances.offsets, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aOffset);
      gl.vertexAttribPointer(this.aOffset, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuf);
      gl.bufferData(gl.ARRAY_BUFFER, pageInstances.sizes, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aSize);
      gl.vertexAttribPointer(this.aSize, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, pageInstances.uvs, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aUV);
      gl.vertexAttribPointer(this.aUV, 4, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.TRIANGLES, 0, pageInstances.count);
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
