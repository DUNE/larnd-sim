// Multi-tile layout 3.0.40, original full ND-LAr module geometry.
// Jsonnet port of multi_tile_layout-3.0.40.yaml.
local base = import 'multi_tile_layout-3.0.40.libsonnet';

local anodeX = 467.88;
local tileZ = 243.2;
// y of each pair of tiles, kept exactly as in the YAML (including its float
// rounding).
local pairY = [
  1348.0,
  1044.0,
  740.0,
  436.0000000000001,
  132.00000000000017,
  -171.9999999999999,
  -475.99999999999994,
  -780.0,
  -1083.9999999999998,
  -1388.0,
];

base + {
  tile:: base.tiles.original,
  drift_length:: null,

  // Tiles come in pairs at the same y: odd slots at z = -tileZ, even slots at
  // z = +tileZ. Anode 1 sits at -x, anode 2 at +x.
  anodes:: [
    {
      x: xSign * anodeX,
      slots: [
        { y: pairY[std.floor((slot - 1) / 2)], z: (if slot % 2 == 1 then -1 else 1) * tileZ, orientation: [-xSign, 1, 1] }
        for slot in std.range(1, 20)
      ],
    }
    for xSign in [-1, 1]
  ],
}
