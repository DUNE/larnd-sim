// Multi-tile layout 2.4.16 (v4). Jsonnet port of multi_tile_layout-2.4.16_v4.yaml.
// Same as v3 with the two anodes on opposite sides.
local base = import 'multi_tile_layout-2.x.libsonnet';

base + {
  multitileLayoutVersion:: '2.4.16',
  swapAnodes:: true,
}
