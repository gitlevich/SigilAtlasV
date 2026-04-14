precision mediump float;

varying vec2 v_texcoord;
uniform sampler2D u_atlas;

void main() {
  gl_FragColor = texture2D(u_atlas, v_texcoord);
}
