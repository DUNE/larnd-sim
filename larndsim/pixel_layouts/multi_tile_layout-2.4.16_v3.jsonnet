// Multi-tile layout 2.4.16 (v3). Jsonnet port of multi_tile_layout-2.4.16_v3.yaml.
// This is the configuration the 2.x base describes by default.
local base = import 'multi_tile_layout-2.x.libsonnet';

base + {
  multitileLayoutVersion:: '2.4.16',
}
