// Multi-tile layout 3.0.40, FSD (v3). Jsonnet port of multi_tile_layout-3.0.40_fsd_v3.yaml.
// The ND-LAr module v3 with each anode on the opposite side in x and its tile
// columns swapped in z (z orientations unchanged).
local base = import 'multi_tile_layout-3.0.40.libsonnet';

base + {
  anodeSides:: [
    { xSign: 1, firstColumnZSign: 1, zOrientation: -1 },
    { xSign: -1, firstColumnZSign: 1, zOrientation: 1 },
  ],
}
