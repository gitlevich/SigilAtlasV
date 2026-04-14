// Per-vertex (unit quad)
attribute vec2 a_position; // 0..1

// Per-instance
attribute vec2 a_offset;   // torus-space x, y
attribute vec2 a_size;     // width, height in torus-space
attribute vec4 a_uv;       // u0, v0, u1, v1 in atlas

// Camera
uniform vec2 u_camera;     // camera x, y in torus-space
uniform float u_zoom;      // visible torus-space width
uniform vec2 u_viewport;   // viewport pixel size
uniform vec2 u_torus;      // torus width, height

varying vec2 v_texcoord;

void main() {
  // Quad corner in torus-space
  vec2 corner = a_offset + a_position * a_size;

  // Camera-relative position with wraparound
  vec2 rel = corner - u_camera;

  // Wrap to nearest representation
  rel.x = mod(rel.x + u_torus.x * 0.5, u_torus.x) - u_torus.x * 0.5;
  rel.y = mod(rel.y + u_torus.y * 0.5, u_torus.y) - u_torus.y * 0.5;

  // Convert to clip space: u_zoom = visible width in torus units
  float aspect = u_viewport.x / u_viewport.y;
  float visibleHeight = u_zoom / aspect;

  vec2 ndc;
  ndc.x = rel.x / (u_zoom * 0.5);
  ndc.y = rel.y / (visibleHeight * 0.5);

  gl_Position = vec4(ndc, 0.0, 1.0);

  // Texture coordinate interpolation
  v_texcoord = mix(a_uv.xy, a_uv.zw, a_position);
}
