/**
 * WebGL torus viewport renderer.
 *
 * Two layout modes:
 *  - Strip mode (@TimeLike): binary-search packed strips, native aspect ratios.
 *  - Grid mode (@SpaceLike): uniform square cells, per-image gravity tweening,
 *    optional @Relief (elevation from density) and @Wireframe overlay.
 *
 * 3D pipeline: cells live at (x, y, z=elevation*reliefScale). The camera
 * orbits around the torus centre with pitch and yaw. At pitch=0 and relief
 * off, the view is identical to the flat 2D previous behaviour.
 *
 * Two-tier resolution: atlas thumbnails batched per page; previews drawn
 * individually on top when rendered taller than PREVIEW_THRESHOLD px.
 * Square crop for @SpaceLike via UV narrowing — shader samples the central
 * square of each thumbnail; no image re-encoding.
 */

import type {
  AnyLayout,
  StripLayout,
  SpaceLikeLayout,
  LayerToggles,
  PointOfView,
  Strip,
} from "../types";
import { isSpaceLikeLayout } from "../types";
import { TextureAtlas, type UVRect } from "./texture-atlas";
import { PreviewCache } from "./preview-cache";
import { OverviewAtlas } from "./overview-atlas";
import { MidAtlas } from "./mid-atlas";

// Four-tier resolution thresholds, in CSS pixels of rendered cell height.
// Overview (15px) is always the base; higher tiers overdraw when available.
const TIER_MID_PX = 30;      // mid-atlas (32px) when cell > this
const TIER_ATLAS_PX = 80;    // streamed per-image atlas (96px) when cell > this
const PREVIEW_THRESHOLD = 200; // 1024px preview when cell > this
const TWEEN_MS = 500;

// 3D world axes: x/y span the torus plane; z is up (elevation).

const VERT_SRC = `
attribute vec2 a_position;
attribute vec3 a_offset;
attribute vec2 a_size;
attribute vec4 a_uv;

uniform mat4 u_mvp;

varying vec2 v_texcoord;

void main() {
  vec3 world = a_offset + vec3(a_position.x * a_size.x, a_position.y * a_size.y, 0.0);
  gl_Position = u_mvp * vec4(world, 1.0);
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

const LINE_VERT_SRC = `
attribute vec3 a_position;
uniform mat4 u_mvp;
void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);
}
`;

const LINE_FRAG_SRC = `
precision mediump float;
uniform vec4 u_color;
void main() {
  gl_FragColor = u_color;
}
`;

const QUAD = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1];

interface PreviewQuad {
  id: string;
  wx: number;
  wy: number;
  wz: number;
  width: number;
  height: number;
  uvU0: number;
  uvV0: number;
  uvU1: number;
  uvV1: number;
}

interface VertexBucket {
  offsets: number[]; // (x, y, z) per vertex
  sizes: number[];
  uvs: number[];
  count: number;
}

interface SpriteTween {
  startCol: number;
  startRow: number;
  targetCol: number;
  targetRow: number;
  tweenStart: number;
}

// ── Matrix math (column-major, 4x4) ──

function mat4Identity(): Float32Array {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function mat4Ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Float32Array {
  const rl = right - left, tb = top - bottom, fn = far - near;
  const m = new Float32Array(16);
  m[0] = 2 / rl; m[5] = 2 / tb; m[10] = -2 / fn; m[15] = 1;
  m[12] = -(right + left) / rl;
  m[13] = -(top + bottom) / tb;
  m[14] = -(far + near) / fn;
  return m;
}

function vec3Sub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function vec3Cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function vec3Norm(v: [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 1e-8 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
}
function vec3Dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function mat4LookAt(eye: [number, number, number], target: [number, number, number], up: [number, number, number]): Float32Array {
  const f = vec3Norm(vec3Sub(target, eye));
  const s = vec3Norm(vec3Cross(f, up));
  const u = vec3Cross(s, f);
  const m = new Float32Array(16);
  m[0] = s[0]; m[4] = s[1]; m[8] = s[2]; m[12] = -vec3Dot(s, eye);
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2]; m[13] = -vec3Dot(u, eye);
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2]; m[14] = vec3Dot(f, eye);
  m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
  return m;
}

export class TorusViewport {
  private gl: WebGLRenderingContext;
  private canvas: HTMLCanvasElement;
  private program: WebGLProgram;
  private lineProgram: WebGLProgram;
  private atlas: TextureAtlas;
  private overview: OverviewAtlas;
  private mid: MidAtlas;
  private previews: PreviewCache;

  private quadBuf: WebGLBuffer;
  private offsetBuf: WebGLBuffer;
  private sizeBuf: WebGLBuffer;
  private uvBuf: WebGLBuffer;
  private lineBuf: WebGLBuffer;

  private aPosition: number;
  private aOffset: number;
  private aSize: number;
  private aUV: number;

  private uMvp: WebGLUniformLocation;
  private uAtlas: WebGLUniformLocation;

  private aLinePosition: number;
  private uLineMvp: WebGLUniformLocation;
  private uLineColor: WebGLUniformLocation;

  private stripLayout: StripLayout | null = null;
  private gridLayout: SpaceLikeLayout | null = null;
  private tweens: Map<string, SpriteTween> = new Map();
  private stripYs: Float64Array = new Float64Array(0);

  // Wireframe edges (spacelike only): pairs of (col, row, id) for each edge
  private wireframeEdges: Array<[number, number, number, number, string, string]> = [];

  // Cached contour geometry per layout, keyed by reliefScale×layers.relief.
  // Regenerated when the elevation grid or relief state changes.
  private contourVerts: Float32Array | null = null;
  private contourReliefScale = -1;
  private contourReliefOn = false;

  // Neighborhood cluster ids per image; cached boundary geometry.
  private clusterIds: Map<string, number> = new Map();
  private boundaryVerts: Float32Array | null = null;

  private thumbnailBaseUrl = "";
  private animFrameId = 0;
  private layers: LayerToggles = { photos: true, neighborhoods: false };
  private reliefScale = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: false, alpha: false, depth: true })!;
    if (!gl) throw new Error("WebGL not supported");
    this.gl = gl;

    this.program = this._createProgram(VERT_SRC, FRAG_SRC);
    this.lineProgram = this._createProgram(LINE_VERT_SRC, LINE_FRAG_SRC);

    gl.useProgram(this.program);
    this.aPosition = gl.getAttribLocation(this.program, "a_position");
    this.aOffset = gl.getAttribLocation(this.program, "a_offset");
    this.aSize = gl.getAttribLocation(this.program, "a_size");
    this.aUV = gl.getAttribLocation(this.program, "a_uv");
    this.uMvp = gl.getUniformLocation(this.program, "u_mvp")!;
    this.uAtlas = gl.getUniformLocation(this.program, "u_atlas")!;

    gl.useProgram(this.lineProgram);
    this.aLinePosition = gl.getAttribLocation(this.lineProgram, "a_position");
    this.uLineMvp = gl.getUniformLocation(this.lineProgram, "u_mvp")!;
    this.uLineColor = gl.getUniformLocation(this.lineProgram, "u_color")!;

    const quad = new Float32Array(QUAD);
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    this.offsetBuf = gl.createBuffer()!;
    this.sizeBuf = gl.createBuffer()!;
    this.uvBuf = gl.createBuffer()!;
    this.lineBuf = gl.createBuffer()!;

    this.atlas = new TextureAtlas(gl);
    this.overview = new OverviewAtlas(gl);
    this.mid = new MidAtlas(gl);
    this.previews = new PreviewCache(gl);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
  }

  setThumbnailBaseUrl(url: string): void { this.thumbnailBaseUrl = url; }

  async loadOverview(): Promise<void> {
    if (!this.thumbnailBaseUrl) return;
    try {
      await this.overview.load(this.thumbnailBaseUrl);
    } catch (e) {
      console.warn("[overview] failed:", e);
    }
  }

  async loadMidAtlas(): Promise<void> {
    if (!this.thumbnailBaseUrl) return;
    try {
      await this.mid.load(this.thumbnailBaseUrl);
    } catch (e) {
      console.warn("[mid-atlas] failed:", e);
    }
  }
  setLayers(layers: LayerToggles): void { this.layers = { ...layers }; }
  setReliefScale(scale: number): void { this.reliefScale = scale; }

  setLayout(layout: AnyLayout): void {
    if (isSpaceLikeLayout(layout)) this._setGridLayout(layout);
    else this._setStripLayout(layout);
  }

  private _setStripLayout(layout: StripLayout): void {
    this.gridLayout = null;
    this.tweens.clear();
    this.wireframeEdges = [];
    this.stripLayout = layout;
    const ys = new Float64Array(layout.strips.length);
    for (let i = 0; i < layout.strips.length; i++) ys[i] = layout.strips[i].y;
    this.stripYs = ys;
  }

  private _setGridLayout(layout: SpaceLikeLayout): void {
    this.stripLayout = null;
    this.gridLayout = layout;
    this.contourVerts = null;  // force regen
    this.boundaryVerts = null;
    const now = performance.now();

    const primaries = new Map<string, { col: number; row: number }>();
    for (const p of layout.positions) {
      if (!primaries.has(p.id)) primaries.set(p.id, { col: p.col, row: p.row });
    }

    const cols = layout.cols, rows = layout.rows;
    const next: Map<string, SpriteTween> = new Map();

    for (const [id, { col, row }] of primaries) {
      const prev = this.tweens.get(id);
      let sCol: number, sRow: number;
      if (prev) {
        const t = Math.min(1, (now - prev.tweenStart) / TWEEN_MS);
        const eased = 0.5 - 0.5 * Math.cos(t * Math.PI);
        sCol = prev.startCol + (prev.targetCol - prev.startCol) * eased;
        sRow = prev.startRow + (prev.targetRow - prev.startRow) * eased;
      } else {
        sCol = col; sRow = row;
      }
      let dCol = col - sCol;
      if (dCol > cols * 0.5) dCol -= cols;
      else if (dCol < -cols * 0.5) dCol += cols;
      let dRow = row - sRow;
      if (dRow > rows * 0.5) dRow -= rows;
      else if (dRow < -rows * 0.5) dRow += rows;

      next.set(id, {
        startCol: sCol, startRow: sRow,
        targetCol: sCol + dCol, targetRow: sRow + dRow,
        tweenStart: now,
      });
    }
    this.tweens = next;
    // Invalidate wireframe edges — caller must re-provide if wanted
    this.wireframeEdges = [];
  }

  setNeighborhoodClusters(clusters: Record<string, number>): void {
    this.clusterIds = new Map(Object.entries(clusters));
    this.boundaryVerts = null;  // regenerate lazily
  }

  setWireframeEdges(edges: Array<[string, string]>): void {
    if (!this.gridLayout) { this.wireframeEdges = []; return; }
    const idToPos = new Map<string, { col: number; row: number; elev: number }>();
    for (const p of this.gridLayout.positions) {
      if (!idToPos.has(p.id)) idToPos.set(p.id, { col: p.col, row: p.row, elev: p.elevation });
    }
    this.wireframeEdges = [];
    for (const [a, b] of edges) {
      const pa = idToPos.get(a), pb = idToPos.get(b);
      if (!pa || !pb) continue;
      this.wireframeEdges.push([pa.col, pa.row, pb.col, pb.row, a, b]);
    }
  }

  // ── MVP construction ──

  private _buildMvp(pov: PointOfView, worldHeightBias: number): Float32Array {
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const w = pov.z;
    const h = pov.z / aspect;

    // Orbit pivot: centre of the content in world space.
    const pivotX = this.gridLayout
      ? (this.gridLayout.torus_width) * 0.5
      : this.stripLayout ? this.stripLayout.torus_width * 0.5 : 0;
    const pivotY = this.gridLayout
      ? (this.gridLayout.torus_height) * 0.5
      : this.stripLayout ? this.stripLayout.torus_height * 0.5 : 0;
    // Hover the pivot above half the relief so tilted views frame terrain nicely.
    const pivotZ = worldHeightBias;

    // Offset the pivot by pan position (pov.x, pov.y are the pan offsets).
    const targetX = pov.x;
    const targetY = pov.y;
    const targetZ = pivotZ;

    // Eye position on a sphere around the target.
    const distance = pov.z;
    const pitch = pov.pitch;
    const yaw = pov.yaw;
    const ex = targetX + distance * Math.sin(yaw) * Math.sin(pitch);
    const ey = targetY - distance * Math.cos(yaw) * Math.sin(pitch);
    const ez = targetZ + distance * Math.cos(pitch);

    // "Up" rotates with yaw and tilts with pitch so it is always perpendicular
    // to the forward direction (the lookAt would degenerate otherwise).
    // At pitch=0: +y is up on screen. At pitch=PI/2: +z is up on screen.
    const upX = -Math.sin(yaw) * Math.cos(pitch);
    const upY = Math.cos(yaw) * Math.cos(pitch);
    const upZ = Math.sin(pitch);

    const near = -distance * 4;
    const far = distance * 4 + this.reliefScale * 2;

    const proj = mat4Ortho(-w / 2, w / 2, -h / 2, h / 2, near, far);
    const view = mat4LookAt([ex, ey, ez], [targetX, targetY, targetZ], [upX, upY, upZ]);
    void pivotX; void pivotY;
    return mat4Multiply(proj, view);
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
    overviewQuads: Map<number, VertexBucket>;
    midPages: Map<number, VertexBucket>;
  } {
    const atlasPages = new Map<number, VertexBucket>();
    const previewQuads: PreviewQuad[] = [];
    const overviewQuads = new Map<number, VertexBucket>();
    const midPages = new Map<number, VertexBucket>();

    const layout = this.stripLayout;
    if (!layout) return { atlasPages, previewQuads, overviewQuads, midPages };
    const { strips, torus_width, torus_height } = layout;
    if (strips.length === 0 || torus_width === 0 || torus_height === 0) return { atlasPages, previewQuads, overviewQuads, midPages };

    const aspect = this.canvas.width / this.canvas.height;
    const visW = pov.z, visH = pov.z / aspect;
    const cssPixelsPerUnit = this.canvas.clientHeight / visH;

    const viewLeft = pov.x - visW * 0.5;
    const viewRight = pov.x + visW * 0.5;
    const viewTop = pov.y - visH * 0.5;
    const viewBottom = pov.y + visH * 0.5;
    const marginX = visW * 0.5, marginY = visH * 0.5;
    const fetchLeft = viewLeft - marginX, fetchRight = viewRight + marginX;
    const fetchTop = viewTop - marginY, fetchBottom = viewBottom + marginY;

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
          const ii0 = this._firstVisibleImage(strip, fetchLeft - dx);
          for (let ii = ii0; ii < strip.images.length; ii++) {
            const img = strip.images[ii];
            if (img.x > fetchRight - dx) break;
            if (img.x + img.width < fetchLeft - dx) continue;
            const wx = img.x + dx;

            if (!this.atlas.isLoaded(img.id) && baseUrl) {
              this.atlas.loadThumbnail(img.id, `${baseUrl}/thumbnail/${img.id}`);
            }
            const inView = stripVisible && wx + img.width >= viewLeft && wx <= viewRight;
            if (!inView) continue;

            if (needsPreview) {
              if (!this.previews.isRequested(img.id) && baseUrl) this.previews.load(img.id, `${baseUrl}/preview/${img.id}`);
              if (this.previews.has(img.id)) {
                previewQuads.push({ id: img.id, wx, wy, wz: 0, width: img.width, height: sh, uvU0: 0, uvV0: 1, uvU1: 1, uvV1: 0 });
                continue;
              }
            }
            const entry = this.atlas.getUV(img.id);
            this._pushQuad(atlasPages, entry.page, wx, wy, 0, img.width, sh, entry.uv.u0, entry.uv.v0, entry.uv.u1, entry.uv.v1);
          }
        }
      }
    }
    return { atlasPages, previewQuads, overviewQuads, midPages };
  }

  // ── Grid mode (SpaceLike) ──

  private _squareCropUV(full: UVRect): UVRect {
    const du = full.u1 - full.u0;
    const dvAbs = Math.abs(full.v0 - full.v1);
    if (du <= 0 || dvAbs <= 0) return full;
    if (du > dvAbs) {
      const uMid = (full.u0 + full.u1) * 0.5, half = dvAbs * 0.5;
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
    overviewQuads: Map<number, VertexBucket>;
    midPages: Map<number, VertexBucket>;
  } {
    const atlasPages = new Map<number, VertexBucket>();
    const previewQuads: PreviewQuad[] = [];
    const overviewQuads = new Map<number, VertexBucket>();
    const midPages = new Map<number, VertexBucket>();

    const layout = this.gridLayout;
    if (!layout) return { atlasPages, previewQuads, overviewQuads, midPages };
    const { positions, cell_size, cols, rows, torus_width, torus_height } = layout;
    if (positions.length === 0) return { atlasPages, previewQuads, overviewQuads, midPages };

    const aspect = this.canvas.width / this.canvas.height;
    const visW = pov.z, visH = pov.z / aspect;
    const cssPixelsPerUnit = this.canvas.clientHeight / visH;
    const renderedH = cell_size * cssPixelsPerUnit;
    const needsPreview = renderedH > PREVIEW_THRESHOLD;
    const needsAtlas = renderedH > TIER_ATLAS_PX;
    const midReady = this.mid.isReady() && renderedH > TIER_MID_PX;

    // Tilt widens the visible strip along the camera's forward direction and
    // elevation projects onto screen-y. Expand bounds to cover the extra world
    // a tilted orthographic view reveals. At pitch=PI/2 the strip is infinite,
    // so we clamp by torus dimensions.
    const pitchFactor = 1 + Math.sin(Math.abs(pov.pitch)) * 4;
    const reliefBleed = false ? this.reliefScale * 2 : 0;
    const viewLeft = pov.x - visW * 0.5;
    const viewRight = pov.x + visW * 0.5;
    const viewTop = pov.y - visH * 0.5 * pitchFactor - reliefBleed;
    const viewBottom = pov.y + visH * 0.5 * pitchFactor + reliefBleed;
    const marginX = visW * 0.5, marginY = visH * 0.5;
    const fetchLeft = viewLeft - marginX, fetchRight = viewRight + marginX;
    const fetchTop = viewTop - marginY, fetchBottom = viewBottom + marginY;

    const baseUrl = this.thumbnailBaseUrl;
    const now = performance.now();
    const seenIds = new Set<string>();
    const reliefOn = false;
    const zScale = reliefOn ? this.reliefScale : 0;

    for (const pos of positions) {
      const isPrimary = !seenIds.has(pos.id);
      seenIds.add(pos.id);
      let curCol: number, curRow: number;
      if (isPrimary) {
        const cp = this._spriteCurrentPos(pos.id, now);
        curCol = cp.col; curRow = cp.row;
      } else {
        curCol = pos.col; curRow = pos.row;
      }
      curCol = ((curCol % cols) + cols) % cols;
      curRow = ((curRow % rows) + rows) % rows;

      const baseWx = curCol * cell_size;
      const baseWy = curRow * cell_size;
      const wz = pos.elevation * zScale;

      let requested = false;
      for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
        const wy = baseWy + dy;
        if (wy + cell_size < fetchTop || wy > fetchBottom) continue;
        for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
          const wx = baseWx + dx;
          if (wx + cell_size < fetchLeft || wx > fetchRight) continue;

          if (needsAtlas && !requested && !this.atlas.isLoaded(pos.id) && baseUrl) {
            this.atlas.loadThumbnail(pos.id, `${baseUrl}/thumbnail/${pos.id}`);
            requested = true;
          }

          const inView = wx + cell_size >= viewLeft && wx <= viewRight && wy + cell_size >= viewTop && wy <= viewBottom;
          if (!inView) continue;

          if (needsPreview) {
            if (!this.previews.isRequested(pos.id) && baseUrl) this.previews.load(pos.id, `${baseUrl}/preview/${pos.id}`);
            if (this.previews.has(pos.id)) {
              const cropped = this._squareCropUV({ u0: 0, v0: 1, u1: 1, v1: 0 });
              previewQuads.push({ id: pos.id, wx, wy, wz, width: cell_size, height: cell_size, uvU0: cropped.u0, uvV0: cropped.v0, uvU1: cropped.u1, uvV1: cropped.v1 });
              continue;
            }
          }
          // Tier selection in descending resolution: preview > atlas > mid >
          // overview. Preview handled above as previewQuads; here we pick
          // between atlas (per-image, streamed) and the pre-baked mid/overview.
          if (this.atlas.hasReal(pos.id)) {
            const entry = this.atlas.getUV(pos.id);
            const cropped = this._squareCropUV(entry.uv);
            this._pushQuad(atlasPages, entry.page, wx, wy, wz, cell_size, cell_size, cropped.u0, cropped.v0, cropped.u1, cropped.v1);
          } else if (midReady) {
            const me = this.mid.getUV(pos.id);
            if (me) {
              this._pushQuad(midPages, me.page, wx, wy, wz, cell_size, cell_size, me.uv.u0, me.uv.v0, me.uv.u1, me.uv.v1);
            } else {
              const ov = this.overview.getUV(pos.id);
              if (ov) this._pushQuad(overviewQuads, 0, wx, wy, wz, cell_size, cell_size, ov.u0, ov.v0, ov.u1, ov.v1);
            }
          } else {
            const ov = this.overview.getUV(pos.id);
            if (ov) this._pushQuad(overviewQuads, 0, wx, wy, wz, cell_size, cell_size, ov.u0, ov.v0, ov.u1, ov.v1);
          }
        }
      }
    }
    return { atlasPages, previewQuads, overviewQuads, midPages };
  }

  private _pushQuad(atlasPages: Map<number, VertexBucket>, page: number, wx: number, wy: number, wz: number, w: number, h: number, u0: number, v0: number, u1: number, v1: number): void {
    let bucket = atlasPages.get(page);
    if (!bucket) { bucket = { offsets: [], sizes: [], uvs: [], count: 0 }; atlasPages.set(page, bucket); }
    for (let v = 0; v < 6; v++) {
      bucket.offsets.push(wx, wy, wz);
      bucket.sizes.push(w, h);
      bucket.uvs.push(u0, v0, u1, v1);
    }
    bucket.count += 6;
  }

  render(pov: PointOfView): void {
    const gl = this.gl;
    const { canvas } = this;

    const dpr = window.devicePixelRatio || 1;
    const displayW = Math.round(canvas.clientWidth * dpr);
    const displayH = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width = displayW; canvas.height = displayH;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (!this.stripLayout && !this.gridLayout) return;

    const worldHeightBias = false ? this.reliefScale * 0.5 : 0;
    const mvp = this._buildMvp(pov, worldHeightBias);

    // Photos pass
    if (this.layers.photos) {
      const { atlasPages, previewQuads, overviewQuads, midPages } = this.stripLayout
        ? this._collectStripVisible(pov)
        : this._collectGridVisible(pov);

      gl.useProgram(this.program);
      gl.uniformMatrix4fv(this.uMvp, false, mvp);

      // Tier stack, lowest to highest: overview → mid → atlas → preview.
      if (this.overview.isReady()) {
        for (const [, bucket] of overviewQuads) {
          if (bucket.count === 0) continue;
          this.overview.bind(0);
          gl.uniform1i(this.uAtlas, 0);
          this._uploadAndDrawQuads(bucket);
        }
      }

      if (this.mid.isReady()) {
        for (const [page, bucket] of midPages) {
          if (bucket.count === 0) continue;
          this.mid.bindPage(page, 0);
          gl.uniform1i(this.uAtlas, 0);
          this._uploadAndDrawQuads(bucket);
        }
      }

      for (const [page, bucket] of atlasPages) {
        if (bucket.count === 0) continue;
        this.atlas.bindPage(page, 0);
        gl.uniform1i(this.uAtlas, 0);
        this._uploadAndDrawQuads(bucket);
      }
      for (const pq of previewQuads) {
        const entry = this.previews.get(pq.id);
        if (!entry) continue;
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, entry.texture);
        gl.uniform1i(this.uAtlas, 0);
        const bucket: VertexBucket = { offsets: [], sizes: [], uvs: [], count: 6 };
        for (let v = 0; v < 6; v++) {
          bucket.offsets.push(pq.wx, pq.wy, pq.wz);
          bucket.sizes.push(pq.width, pq.height);
          bucket.uvs.push(pq.uvU0, pq.uvV0, pq.uvU1, pq.uvV1);
        }
        this._uploadAndDrawQuads(bucket);
      }
    }

    // Neighborhoods overlay — cluster boundary lines.
    if (this.layers.neighborhoods && this.gridLayout && this.clusterIds.size > 0) {
      this._drawNeighborhoodBoundaries(mvp, pov);
    }
  }

  // ── Neighborhood boundaries ──

  private _buildBoundaryVerts(): Float32Array {
    const layout = this.gridLayout!;
    const { positions, cell_size, cols, rows } = layout;

    // Build cols×rows grid of cluster ids. -1 = unassigned.
    const grid = new Int32Array(cols * rows);
    grid.fill(-1);
    for (const p of positions) {
      const cid = this.clusterIds.get(p.id);
      if (cid !== undefined) grid[p.row * cols + p.col] = cid;
    }

    const verts: number[] = [];
    // For each cell, draw a line on its right edge if the cell to the right
    // has a different cluster id; same for its top edge. Torus-wrap adjacency.
    for (let r = 0; r < rows; r++) {
      const rUp = (r + 1) % rows;
      for (let c = 0; c < cols; c++) {
        const cRight = (c + 1) % cols;
        const here = grid[r * cols + c];
        if (here < 0) continue;

        const right = grid[r * cols + cRight];
        if (right >= 0 && right !== here) {
          const x = (c + 1) * cell_size;
          const y0 = r * cell_size;
          const y1 = (r + 1) * cell_size;
          verts.push(x, y0, 0, x, y1, 0);
        }
        const up = grid[rUp * cols + c];
        if (up >= 0 && up !== here) {
          const y = (r + 1) * cell_size;
          const x0 = c * cell_size;
          const x1 = (c + 1) * cell_size;
          verts.push(x0, y, 0, x1, y, 0);
        }
      }
    }
    return new Float32Array(verts);
  }

  private _drawNeighborhoodBoundaries(mvp: Float32Array, pov: PointOfView): void {
    const gl = this.gl;
    const layout = this.gridLayout!;
    if (!this.boundaryVerts) {
      this.boundaryVerts = this._buildBoundaryVerts();
    }
    if (this.boundaryVerts.length === 0) return;

    const { torus_width, torus_height } = layout;
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const visW = pov.z, visH = pov.z / aspect;
    const viewLeft = pov.x - visW, viewRight = pov.x + visW;
    const viewTop = pov.y - visH, viewBottom = pov.y + visH;

    // Tile across the 9 wrap positions; cull tile copies outside the view.
    const tiled: number[] = [];
    const src = this.boundaryVerts;
    for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
      if (dy + torus_height < viewTop || dy > viewBottom) continue;
      for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
        if (dx + torus_width < viewLeft || dx > viewRight) continue;
        for (let i = 0; i < src.length; i += 3) {
          tiled.push(src[i] + dx, src[i + 1] + dy, src[i + 2]);
        }
      }
    }
    if (tiled.length === 0) return;

    gl.useProgram(this.lineProgram);
    gl.uniformMatrix4fv(this.uLineMvp, false, mvp);
    gl.uniform4f(this.uLineColor, 0.85, 0.95, 1.0, 0.75);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tiled), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aLinePosition);
    gl.vertexAttribPointer(this.aLinePosition, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, tiled.length / 3);
  }

  // ── Contours ──

  private _buildContourVerts(): Float32Array {
    const layout = this.gridLayout!;
    const { positions, cell_size, cols, rows } = layout;

    // Fill a cols×rows grid with elevation from (primary) positions.
    // Duplicates (from padding) overwrite silently; their elevation is the same.
    const grid = new Float32Array(cols * rows);
    for (const p of positions) {
      grid[p.row * cols + p.col] = p.elevation;
    }

    const zScale = false ? this.reliefScale : 0;
    const levels: number[] = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9];
    const verts: number[] = [];

    // Marching squares, torus-wrapping in both directions.
    for (let r = 0; r < rows; r++) {
      const r1 = (r + 1) % rows;
      for (let c = 0; c < cols; c++) {
        const c1 = (c + 1) % cols;
        const e00 = grid[r * cols + c];
        const e10 = grid[r * cols + c1];
        const e11 = grid[r1 * cols + c1];
        const e01 = grid[r1 * cols + c];

        // Corner world positions (centres offset by half-cell so contour
        // runs through cell centres — standard marching-squares convention).
        const x0 = (c + 0.5) * cell_size;
        const x1 = (c + 1.5) * cell_size;
        const y0 = (r + 0.5) * cell_size;
        const y1 = (r + 1.5) * cell_size;

        for (const lvl of levels) {
          const b0 = e00 >= lvl ? 1 : 0;
          const b1 = e10 >= lvl ? 2 : 0;
          const b2 = e11 >= lvl ? 4 : 0;
          const b3 = e01 >= lvl ? 8 : 0;
          const mask = b0 | b1 | b2 | b3;
          if (mask === 0 || mask === 15) continue;

          // Edge interpolation: linear, clamped.
          const lerp = (a: number, b: number) => {
            const d = b - a;
            if (Math.abs(d) < 1e-9) return 0.5;
            return Math.max(0, Math.min(1, (lvl - a) / d));
          };
          // Four edges: bottom (c,r)-(c+1,r), right (c+1,r)-(c+1,r+1),
          //             top (c+1,r+1)-(c,r+1), left (c,r+1)-(c,r).
          const tB = lerp(e00, e10), pB: [number, number] = [x0 + (x1 - x0) * tB, y0];
          const tR = lerp(e10, e11), pR: [number, number] = [x1, y0 + (y1 - y0) * tR];
          const tT = lerp(e01, e11), pT: [number, number] = [x0 + (x1 - x0) * tT, y1];
          const tL = lerp(e00, e01), pL: [number, number] = [x0, y0 + (y1 - y0) * tL];

          // 16-case line table. Produces line segments between interpolated
          // edge crossings. z at each endpoint = contour level × zScale so
          // contours lift with the terrain under relief.
          const z = lvl * zScale;
          const push = (a: [number, number], b: [number, number]) => {
            verts.push(a[0], a[1], z, b[0], b[1], z);
          };
          switch (mask) {
            case 1: case 14: push(pB, pL); break;
            case 2: case 13: push(pB, pR); break;
            case 3: case 12: push(pL, pR); break;
            case 4: case 11: push(pR, pT); break;
            case 6: case 9:  push(pB, pT); break;
            case 7: case 8:  push(pL, pT); break;
            case 5:          push(pL, pT); push(pB, pR); break; // saddle
            case 10:         push(pL, pB); push(pR, pT); break; // saddle
          }
        }
      }
    }
    return new Float32Array(verts);
  }

  private _drawContours(mvp: Float32Array, pov: PointOfView): void {
    const gl = this.gl;
    const layout = this.gridLayout!;
    const reliefOn = false;

    if (!this.contourVerts || this.contourReliefScale !== this.reliefScale || this.contourReliefOn !== reliefOn) {
      this.contourVerts = this._buildContourVerts();
      this.contourReliefScale = this.reliefScale;
      this.contourReliefOn = reliefOn;
    }
    if (this.contourVerts.length === 0) return;

    const { torus_width, torus_height } = layout;
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const visW = pov.z, visH = pov.z / aspect;
    const pitchFactor = 1 + Math.sin(Math.abs(pov.pitch)) * 4;
    const reliefBleed = reliefOn ? this.reliefScale * 2 : 0;
    const viewLeft = pov.x - visW * 0.5 - visW * 0.5;
    const viewRight = pov.x + visW * 0.5 + visW * 0.5;
    const viewTop = pov.y - visH * 0.5 * pitchFactor - reliefBleed - visH * 0.5;
    const viewBottom = pov.y + visH * 0.5 * pitchFactor + reliefBleed + visH * 0.5;

    // Tile the same vertex array across torus wraps using the existing shader.
    gl.useProgram(this.lineProgram);
    gl.uniformMatrix4fv(this.uLineMvp, false, mvp);
    gl.uniform4f(this.uLineColor, 0.75, 0.85, 0.95, 0.35);

    // Build tiled verts for visible wrap copies only.
    const tiled: number[] = [];
    const src = this.contourVerts;
    for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
      for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
        if (dx + torus_width < viewLeft || dx > viewRight) continue;
        if (dy + torus_height < viewTop || dy > viewBottom) continue;
        for (let i = 0; i < src.length; i += 3) {
          tiled.push(src[i] + dx, src[i + 1] + dy, src[i + 2]);
        }
      }
    }
    if (tiled.length === 0) return;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tiled), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aLinePosition);
    gl.vertexAttribPointer(this.aLinePosition, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, tiled.length / 3);
  }

  private _drawWireframe(mvp: Float32Array, pov: PointOfView): void {
    const gl = this.gl;
    const layout = this.gridLayout!;
    const { cell_size, cols, rows, torus_width, torus_height } = layout;
    const zScale = false ? this.reliefScale : 0;

    const idToPos = new Map<string, { col: number; row: number; elev: number }>();
    for (const p of layout.positions) if (!idToPos.has(p.id)) idToPos.set(p.id, { col: p.col, row: p.row, elev: p.elevation });

    // View bounds (generous, for per-tile cull)
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const visW = pov.z, visH = pov.z / aspect;
    const pitchFactor = 1 + Math.sin(Math.abs(pov.pitch)) * 4;
    const reliefBleed = false ? this.reliefScale * 2 : 0;
    const viewLeft = pov.x - visW * 0.5;
    const viewRight = pov.x + visW * 0.5;
    const viewTop = pov.y - visH * 0.5 * pitchFactor - reliefBleed;
    const viewBottom = pov.y + visH * 0.5 * pitchFactor + reliefBleed;

    const verts: number[] = [];
    // Tile each edge across the 9 torus-wrap offsets so the mesh reads as
    // continuous. Per-tile bounding-box cull keeps the vertex count bounded.
    for (const [, , , , idA, idB] of this.wireframeEdges) {
      const a = idToPos.get(idA);
      const b = idToPos.get(idB);
      if (!a || !b) continue;

      let dc = b.col - a.col;
      if (dc > cols * 0.5) dc -= cols; else if (dc < -cols * 0.5) dc += cols;
      let dr = b.row - a.row;
      if (dr > rows * 0.5) dr -= rows; else if (dr < -rows * 0.5) dr += rows;

      const ax = (a.col + 0.5) * cell_size;
      const ay = (a.row + 0.5) * cell_size;
      const az = a.elev * zScale;
      const bx = ax + dc * cell_size;
      const by = ay + dr * cell_size;
      const bz = b.elev * zScale;

      const edgeMinX = Math.min(ax, bx), edgeMaxX = Math.max(ax, bx);
      const edgeMinY = Math.min(ay, by), edgeMaxY = Math.max(ay, by);

      for (let dy = -torus_height; dy <= torus_height; dy += torus_height) {
        if (edgeMaxY + dy < viewTop || edgeMinY + dy > viewBottom) continue;
        for (let dx = -torus_width; dx <= torus_width; dx += torus_width) {
          if (edgeMaxX + dx < viewLeft || edgeMinX + dx > viewRight) continue;
          verts.push(ax + dx, ay + dy, az, bx + dx, by + dy, bz);
        }
      }
    }
    if (verts.length === 0) return;

    gl.useProgram(this.lineProgram);
    gl.uniformMatrix4fv(this.uLineMvp, false, mvp);
    gl.uniform4f(this.uLineColor, 0.55, 0.8, 1.0, 0.35);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aLinePosition);
    gl.vertexAttribPointer(this.aLinePosition, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, verts.length / 3);
  }

  private _uploadAndDrawQuads(bucket: VertexBucket): void {
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
    gl.vertexAttribPointer(this.aOffset, 3, gl.FLOAT, false, 0, 0);

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

  stopRenderLoop(): void { cancelAnimationFrame(this.animFrameId); }

  private _createProgram(vsrc: string, fsrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = this._compileShader(gl.VERTEX_SHADER, vsrc);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("Program link error: " + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  private _compileShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error("Shader compile error: " + gl.getShaderInfoLog(s));
    }
    return s;
  }
}
