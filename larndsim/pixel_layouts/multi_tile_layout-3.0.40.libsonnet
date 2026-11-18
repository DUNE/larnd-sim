// Shared base for the 3.0.40 multi-tile layouts: ND-LAr modules with two
// anodes of 20 tiles each, every tile a 10-row x 16-column grid of chips with
// 8x8 channels.
//
// Concrete layouts extend this object and override the hidden (::)
// parameters below. The defaults describe
// multi_tile_layout-3.0.40_NDLArModule_v3.

local util = import 'layout_utils.libsonnet';

{
  // ------------------------------------------------------------------------
  // Reference data
  // ------------------------------------------------------------------------

  // Two variants of tile design 3.0.0. Chips are numbered down each column,
  // then column by column from left to right. channelGrid is the channel
  // layout shared by every chip (first row = top of the chip); ioGrid is each
  // chip's io channel (1-4) within the tile.
  tiles:: {
    // As in the original multi_tile_layout-3.0.40.yaml.
    original: {
      version: '3.0.0',
      pixelPitch: 3.8,
      firstChip: 10,
      channelGrid: [
        [59, 60, 61, 63, 0, 2, 3, 4],
        [58, 57, 56, 62, 1, 6, 7, 5],
        [53, 55, 54, 52, 10, 8, 9, 11],
        [49, 51, 50, 48, 14, 12, 13, 15],
        [47, 45, 46, 44, 16, 18, 17, 19],
        [43, 41, 40, 42, 20, 22, 23, 21],
        [37, 38, 39, 33, 30, 24, 25, 26],
        [36, 35, 34, 32, 31, 29, 28, 27],
      ],
      ioGrid: [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      ],
    },
    // As in the *_v3 and fsdcube layouts.
    v3: {
      version: '3.0.0',
      pixelPitch: 3.72,
      firstChip: 11,
      channelGrid: [
        [59, 58, 53, 49, 47, 43, 37, 36],
        [60, 57, 55, 51, 45, 41, 38, 35],
        [61, 56, 54, 50, 46, 40, 39, 34],
        [63, 62, 52, 48, 44, 42, 33, 32],
        [0, 1, 10, 14, 16, 20, 30, 31],
        [2, 6, 8, 12, 18, 22, 24, 29],
        [3, 7, 9, 13, 17, 23, 25, 28],
        [4, 5, 11, 15, 19, 21, 26, 27],
      ],
      ioGrid: [
        [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      ],
    },
  },

  // ------------------------------------------------------------------------
  // Parameters
  // ------------------------------------------------------------------------

  tile:: self.tiles.v3,
  chipIds:: util.chipIdGrid(self.tile.firstChip, rows=10, cols=16, columnMajor=true),

  // io: every 10 consecutive tiles share an io_group.
  tilesPerIoGroup:: 10,

  // Geometry (mm). Each anode has two columns of 10 tiles: slots 1-10 run
  // down rowY at z = firstColumnZSign * tileZ, slots 11-20 at the opposite z.
  // An anode at x = xSign * anodeX drifts towards -xSign; its tiles have
  // orientation [-xSign, 1, zOrientation].
  anodeX:: 471.055,
  rowY:: [1339.2, 1041.6, 744.0, 446.4, 148.8, -148.8, -446.4, -744.0, -1041.6, -1339.2],
  tileZ:: 238.08,
  anodeSides:: [
    { xSign: -1, firstColumnZSign: -1, zOrientation: -1 },
    { xSign: 1, firstColumnZSign: -1, zOrientation: 1 },
  ],

  anodes::
    local p = self;
    [
      {
        x: side.xSign * p.anodeX,
        slots: [
          {
            y: y,
            z: zSign * side.firstColumnZSign * p.tileZ,
            orientation: [-side.xSign, 1, side.zOrientation],
          }
          for zSign in [1, -1]
          for y in p.rowY
        ],
      }
      for side in p.anodeSides
    ],

  // ------------------------------------------------------------------------
  // Output
  // ------------------------------------------------------------------------

  multitile_layout_version: '3.0.40',
  tile_layout_version: self.tile.version,
  pixel_pitch: self.tile.pixelPitch,
  // mm. Layouts without a drift length hide it with `drift_length:: null`.
  drift_length: 467.88,

  local nTiles = std.sum([std.length(a.slots) for a in self.anodes]),
  chip_channel_to_position: util.chipChannelToPosition(self.chipIds, self.tile.channelGrid),
  tile_chip_to_io: util.tileChipToIo(self.chipIds, self.tile.ioGrid, nTiles, self.tilesPerIoGroup),

  local tables = util.tileTables(self.anodes),
  tile_indeces: tables.tile_indeces,
  tile_orientations: tables.tile_orientations,
  tile_positions: tables.tile_positions,
}
