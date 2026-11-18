// Multi-tile layout 2.0.16. Jsonnet port of multi_tile_layout-2.0.16.yaml.
local base = import 'multi_tile_layout-2.x.libsonnet';

base + {
  multitileLayoutVersion:: '2.0.16',
  drift_length:: null,

  // One io_group per tile.
  tilesPerIoGroup:: 1,
  ioGrid:: base.ioGrids['2.0.16'],

  anodeX:: 315.1745,
  pairY:: [465.2, 155.2, -155.2, -465.2],
  tileZ:: 155.2,
  mirrorAnode2Z:: false,
  tileIndexPrefix:: [1],
  tpcCenters:: { '1': [0, -218.236, 0], '2': [0, -218.236, 0] },
}
