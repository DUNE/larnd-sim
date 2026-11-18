// Builders shared by the multi-tile pixel layout files.
//
// Grids describe a tile as seen from the front: grid[r][c] has r = 0 as the
// top row (largest y) and c = 0 as the left column (smallest x).
//
// JSON object keys are always strings, so ids that are integer keys in the
// YAML layouts (chip*1000+channel, chip, tile, tpc) are strings here.

local str = std.toString;
local indices(arr) = std.range(0, std.length(arr) - 1);

{
  // Grid of chip ids, numbered from firstChip either along rows (left to
  // right, then top to bottom) or along columns (top to bottom, then left to
  // right).
  chipIdGrid(firstChip, rows, cols, columnMajor=false):: [
    [
      if columnMajor then firstChip + c * rows + r else firstChip + r * cols + c
      for c in std.range(0, cols - 1)
    ]
    for r in std.range(0, rows - 1)
  ],

  // chip_channel_to_position: every chip has the same square channelGrid
  // (channel numbers laid out like the chip's pixels) and the chips tile the
  // plane as in chipIds. Positions are in units of pixels from the bottom-left.
  chipChannelToPosition(chipIds, channelGrid)::
    local n = std.length(channelGrid);
    local rows = std.length(chipIds);
    {
      [str(chipIds[r][c] * 1000 + channelGrid[i][j])]: [
        c * n + j,
        (rows - 1 - r) * n + (n - 1 - i),
      ]
      for r in indices(chipIds)
      for c in indices(chipIds[r])
      for i in indices(channelGrid)
      for j in indices(channelGrid[i])
    },

  // tile_chip_to_io: value is io_group * 1000 + io_channel.
  //
  // Consecutive blocks of tilesPerIoGroup tiles share an io_group; within a
  // group each tile owns 4 io channels. ioGrid gives each chip's io channel
  // (1-4) within its tile. overrides maps a tile id to a replacement ioGrid
  // for that tile; null entries in a grid mean the chip is not connected.
  tileChipToIo(chipIds, ioGrid, nTiles, tilesPerIoGroup, overrides={}):: {
    [str(tile)]:
      local grid = std.get(overrides, str(tile), ioGrid);
      local ioGroup = std.floor((tile - 1) / tilesPerIoGroup) + 1;
      local ioChannelOffset = 4 * ((tile - 1) % tilesPerIoGroup);
      {
        [str(chipIds[r][c])]: ioGroup * 1000 + ioChannelOffset + grid[r][c]
        for r in indices(chipIds)
        for c in indices(chipIds[r])
        if grid[r][c] != null
      }
    for tile in std.range(1, nTiles)
  },

  // Copy of grid with the given chips marked as not connected (null).
  withoutChips(chipIds, grid, chips):: [
    [
      if std.member(chips, chipIds[r][c]) then null else grid[r][c]
      for c in indices(grid[r])
    ]
    for r in indices(grid)
  ],

  // tile_indeces, tile_orientations and tile_positions.
  //
  // anodes: [{ x, slots: [{ y, z, orientation }] }]. Tiles are numbered from 1,
  // going through the slots of anode 1, then anode 2, and so on. Each tile's
  // index is indexPrefix + [anode, slot].
  tileTables(anodes, indexPrefix=[])::
    local tiles = [
      { anode: a + 1, slot: s + 1, x: anodes[a].x } + anodes[a].slots[s]
      for a in indices(anodes)
      for s in indices(anodes[a].slots)
    ];
    {
      tile_indeces: {
        [str(i + 1)]: indexPrefix + [tiles[i].anode, tiles[i].slot]
        for i in indices(tiles)
      },
      tile_orientations: {
        [str(i + 1)]: tiles[i].orientation
        for i in indices(tiles)
      },
      tile_positions: {
        [str(i + 1)]: [tiles[i].x, tiles[i].y, tiles[i].z]
        for i in indices(tiles)
      },
    },
}
