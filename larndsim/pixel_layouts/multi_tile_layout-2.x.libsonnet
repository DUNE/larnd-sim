// Shared base for the 2.x multi-tile layouts: one module with two anodes of
// 8 tiles each, every tile a 10x10 grid of chips.
//
// Concrete layouts extend this object and override the hidden (::)
// parameters below. The defaults describe multi_tile_layout-2.4.16_v3.

local util = import 'layout_utils.libsonnet';

local chipIds = util.chipIdGrid(firstChip=11, rows=10, cols=10);

{
  // ------------------------------------------------------------------------
  // Reference data
  // ------------------------------------------------------------------------

  chipIds:: chipIds,

  // Tile designs: pixel pitch (mm) and the channel layout shared by every
  // chip (first row = top of the chip).
  tiles:: {
    '2.4.0': {
      version: '2.4.0',
      pixelPitch: 4.434,
      channelGrid: [
        [3, 2, 1, 63, 62, 61, 60],
        [10, 5, 4, 0, 59, 58, 52],
        [13, 12, 11, 49, 50, 51, 53],
        [17, 16, 15, 14, 46, 47, 48],
        [20, 21, 18, 42, 43, 44, 45],
        [19, 26, 27, 32, 36, 37, 41],
        [28, 29, 30, 31, 33, 34, 35],
      ],
    },
    '2.5.1': {
      version: '2.5.1',
      pixelPitch: 3.87975,
      channelGrid: [
        [4, 3, 2, 0, 63, 61, 60, 59],
        [5, 7, 6, 1, 62, 56, 57, 58],
        [11, 9, 8, 10, 52, 54, 55, 53],
        [15, 13, 12, 14, 48, 50, 51, 49],
        [19, 17, 18, 16, 44, 46, 45, 47],
        [21, 23, 22, 20, 42, 40, 41, 43],
        [26, 25, 24, 30, 33, 39, 38, 37],
        [27, 28, 29, 31, 32, 34, 35, 36],
      ],
    },
  },

  // io channel (1-4) of each chip within its tile, laid out like chipIds.
  ioGrids:: {
    // Used up to 2.3.16.
    '2.0.16': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [4, 4, 4, 4, 3, 3, 3, 3, 3, 3],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ],
    '2.4.16': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ],
  },

  // Per-tile io grids for the tiles of Module-0 (2.1.16 - 2.3.16) whose
  // wiring differs from ioGrids['2.0.16']. null = chip not connected.
  module0IoOverrides:: {
    '4': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ],
    '6': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
      [2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 2, 2, 2, 2, 2, 2, 2, 2, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ],
    '7': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ],
    '10': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [4, 4, 4, 4, 3, 3, 3, 3, 3, 3],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ],
    '11': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 4, 3, 3, 3, 3, 3, 3, 3, 3],
      [4, 4, 3, 3, 3, 3, 3, 3, 3, 3],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ],
    '15': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
      [null, 2, 2, 2, 2, 2, 2, 2, 1, 1],
      [2, 2, 2, 2, 2, 2, 2, 2, null, null],
      [2, 2, 2, 2, 2, 2, 2, 2, null, null],
      [null, null, null, null, null, null, null, null, null, null],
    ],
    '16': [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [4, 4, 4, 4, 3, 3, 3, 3, 3, 3],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ],
  },

  // ------------------------------------------------------------------------
  // Parameters
  // ------------------------------------------------------------------------

  multitileLayoutVersion:: '2.4.16',
  tile:: self.tiles['2.4.0'],

  // io: tiles 1-8 are io_group 1 and tiles 9-16 io_group 2 (2.0.16 used one
  // io_group per tile, i.e. tilesPerIoGroup = 1).
  tilesPerIoGroup:: 8,
  ioGrid:: self.ioGrids['2.4.16'],
  ioOverrides:: {},

  // Geometry (mm). Anode 1 sits at -anodeX and drifts towards +x, anode 2 at
  // +anodeX drifting towards -x; swapAnodes exchanges the two sides.
  anodeX:: 305.8975,
  swapAnodes:: false,
  // Tiles come in pairs at the same y: odd slots at z = -tileZ, even slots at
  // z = +tileZ. mirrorAnode2Z flips z (and the z orientation) on anode 2.
  pairY:: [465.57, 155.19, -155.19, -465.57],
  tileZ:: 155.19,
  mirrorAnode2Z:: true,
  // 2.0.16 - 2.2.16 index tiles as [1, anode, slot] rather than [anode, slot].
  tileIndexPrefix:: [],
  tpcCenters:: { '1': [0, 0, 0], '2': [0, 0, 0] },

  anodes::
    local p = self;
    [
      local xSign = (if anode == 1 then -1 else 1) * (if p.swapAnodes then -1 else 1);
      local zFlip = if anode == 2 && p.mirrorAnode2Z then -1 else 1;
      {
        x: xSign * p.anodeX,
        slots: [
          local odd = slot % 2 == 1;
          {
            y: p.pairY[std.floor((slot - 1) / 2)],
            z: zFlip * (if odd then -1 else 1) * p.tileZ,
            orientation: [-xSign, if odd then -1 else 1, zFlip * (if odd then 1 else -1)],
          }
          for slot in std.range(1, 8)
        ],
      }
      for anode in [1, 2]
    ],

  // ------------------------------------------------------------------------
  // Output
  // ------------------------------------------------------------------------

  multitile_layout_version: self.multitileLayoutVersion,
  tile_layout_version: self.tile.version,
  pixel_pitch: self.tile.pixelPitch,
  // mm. Layouts without a drift length hide it with `drift_length:: null`.
  drift_length: 304.31,

  chip_channel_to_position: util.chipChannelToPosition(chipIds, self.tile.channelGrid),
  tile_chip_to_io: util.tileChipToIo(chipIds, self.ioGrid, 16, self.tilesPerIoGroup, self.ioOverrides),

  local tables = util.tileTables(self.anodes, self.tileIndexPrefix),
  tile_indeces: tables.tile_indeces,
  tile_orientations: tables.tile_orientations,
  tile_positions: tables.tile_positions,
  tpc_centers: self.tpcCenters,
}
