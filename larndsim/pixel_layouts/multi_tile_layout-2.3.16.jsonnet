// Multi-tile layout 2.3.16 (Module-0, multi-module convention).
// Jsonnet port of multi_tile_layout-2.3.16.yaml.
local base = import 'multi_tile_layout-2.x.libsonnet';

base + {
  multitileLayoutVersion:: '2.3.16',
  drift_length:: null,

  ioGrid:: base.ioGrids['2.0.16'],
  ioOverrides:: base.module0IoOverrides,

  anodeX:: 304.31,
}
