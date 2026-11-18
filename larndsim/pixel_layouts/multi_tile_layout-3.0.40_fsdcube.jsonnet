// Multi-tile layout 3.0.40, FSD cube. Jsonnet port of multi_tile_layout-3.0.40_fsdcube.yaml.
// A single tile: FSD tile 1, moved to x = 480.675.
local base = import 'multi_tile_layout-3.0.40.libsonnet';

base + {
  drift_length: 477.5,

  anodes:: [
    { x: 480.675, slots: [{ y: 1339.2, z: 238.08, orientation: [-1, 1, -1] }] },
  ],
}
