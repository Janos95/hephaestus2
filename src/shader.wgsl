struct VertexOut {
  @location(2) color : vec4<f32>,
  @builtin(position) position : vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos : vec2<f32>, @location(1) color : vec4<f32>)
                       -> VertexOut {
  var output : VertexOut;
  output.position = vec4<f32>(pos,0.,1.);
  output.color = color;
  return output;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
