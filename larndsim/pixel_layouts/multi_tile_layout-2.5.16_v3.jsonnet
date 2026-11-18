// Multi-tile layout 2.5.16 (v3). Jsonnet port of multi_tile_layout-2.5.16_v3.yaml.
// 2.4.16_v3 with the 8x8-channel tile design 2.5.1.
local base = import 'multi_tile_layout-2.x.libsonnet';

base + {
  multitileLayoutVersion:: '2.5.16',
  tile:: base.tiles['2.5.1'],
}
