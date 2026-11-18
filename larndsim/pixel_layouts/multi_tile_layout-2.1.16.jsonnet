// Multi-tile layout 2.1.16 (Module-0). Jsonnet port of multi_tile_layout-2.1.16.yaml.
local base = import 'multi_tile_layout-2.x.libsonnet';
local util = import 'layout_utils.libsonnet';

base + {
  multitileLayoutVersion:: '2.1.16',
  drift_length:: null,

  ioGrid:: base.ioGrids['2.0.16'],
  // Module-0 wiring, with chip 12 on tile 7 not connected.
  ioOverrides:: base.module0IoOverrides + {
    '7': util.withoutChips(base.chipIds, base.module0IoOverrides['7'], [12]),
  },

  anodeX:: 315.1745,
  pairY:: [465.2, 155.2, -155.2, -465.2],
  tileZ:: 155.2,
  tileIndexPrefix:: [1],
}
