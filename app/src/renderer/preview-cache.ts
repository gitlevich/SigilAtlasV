/**
 * LRU cache of high-resolution preview textures.
 *
 * Unlike the thumbnail atlas (many small images packed into shared textures),
 * previews are individual GL textures — large (1024px), few visible at once,
 * evicted when scrolled away.
 *
 * Concurrency-limited fetching, same pattern as texture-atlas.ts.
 */

const MAX_ENTRIES = 50;
const MAX_CONCURRENT = 6;

interface CacheEntry {
  texture: WebGLTexture;
  width: number;
  height: number;
  lastUsed: number;
}

export class PreviewCache {
  private gl: WebGLRenderingContext;
  private entries: Map<string, CacheEntry> = new Map();
  private loadingSet: Set<string> = new Set();
  private inFlight = 0;
  private queue: Array<{ id: string; url: string }> = [];
  private tick = 0;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
  }

  /** Check if a preview is loaded and ready to draw. */
  has(id: string): boolean {
    return this.entries.has(id);
  }

  /** Check if a preview is loading or loaded. */
  isRequested(id: string): boolean {
    return this.entries.has(id) || this.loadingSet.has(id);
  }

  /** Get the texture for a loaded preview. Marks it as recently used. */
  get(id: string): { texture: WebGLTexture; width: number; height: number } | null {
    const entry = this.entries.get(id);
    if (!entry) return null;
    entry.lastUsed = ++this.tick;
    return entry;
  }

  /** Request a preview load. No-op if already loaded or in flight. */
  load(id: string, url: string): void {
    if (this.entries.has(id) || this.loadingSet.has(id)) return;
    this.loadingSet.add(id);

    if (this.inFlight < MAX_CONCURRENT) {
      this._startFetch(id, url);
    } else {
      this.queue.push({ id, url });
    }
  }

  private _startFetch(id: string, url: string): void {
    this.inFlight++;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      this._onLoaded(id, img);
      this._drainQueue();
    };
    img.onerror = () => {
      this.loadingSet.delete(id);
      this.inFlight--;
      this._drainQueue();
    };
    img.src = url;
  }

  private _onLoaded(id: string, img: HTMLImageElement): void {
    this.loadingSet.delete(id);
    this.inFlight--;

    // Evict if over capacity
    while (this.entries.size >= MAX_ENTRIES) {
      this._evictLRU();
    }

    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.entries.set(id, {
      texture: tex,
      width: img.naturalWidth,
      height: img.naturalHeight,
      lastUsed: ++this.tick,
    });
  }

  private _evictLRU(): void {
    let oldest: string | null = null;
    let oldestTick = Infinity;
    for (const [id, entry] of this.entries) {
      if (entry.lastUsed < oldestTick) {
        oldestTick = entry.lastUsed;
        oldest = id;
      }
    }
    if (oldest) {
      const entry = this.entries.get(oldest)!;
      this.gl.deleteTexture(entry.texture);
      this.entries.delete(oldest);
    }
  }

  private _drainQueue(): void {
    while (this.inFlight < MAX_CONCURRENT && this.queue.length > 0) {
      const next = this.queue.shift()!;
      if (this.entries.has(next.id)) {
        this.loadingSet.delete(next.id);
        continue;
      }
      this._startFetch(next.id, next.url);
    }
  }
}
