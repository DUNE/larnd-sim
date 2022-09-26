import numpy as np
from math import ceil

class TrackSegmentBatcher(object):
    """ Base class for LArND-sim simulation batching, implements an iterator that creates masks into an array of track segments """
    
    def __init__(self, track_seg, **kwargs):
        self.track_seg = track_seg

    def __iter__(self):
        raise NotImplementedError

class TPCBatcher(TrackSegmentBatcher):
    """ Batch generator that generates batches containing a single (or multiple) TPCs per iteration """

    def __init__(self, track_seg, tpc_batch_size=1, tpc_borders=np.empty((0,3,2), dtype='f4')):
        super().__init__(track_seg)

        self.tpc_batch_size = tpc_batch_size
        self.tpc_borders = tpc_borders
        
        self._simulated = np.zeros_like(self.track_seg['trackID'], dtype=bool)
        self._events = np.unique(self.track_seg['eventID'])
        self._curr_event = 0
        self._curr_tpc = 0

        #print(f"TPCBatcher(tracks={self.track_seg.shape}, batch_size={self.tpc_batch_size}, events={len(self._events)}, curr_event={self._curr_event}, curr_tpc={self._curr_tpc})")

    def __len__(self):
        return len(self._events) * ceil(self.tpc_borders.shape[0] / self.tpc_batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        #print(f"TPCBatcher.next(tracks={self.track_seg.shape}, batch_size={self.tpc_batch_size}, events={len(self._events)}, curr_event={self._curr_event}, curr_tpc={self._curr_tpc}, simulated={self._simulated.sum()})")
        
        # if all TPCs have been simulated, continue to next event
        if self._curr_tpc >= self.tpc_borders.shape[0]:
            self._curr_event += 1
            self._curr_tpc = 0
            
        # if all events have been simulated, stop
        if self._curr_event >= len(self._events):
            raise StopIteration
        
        mask = ~self._simulated.copy()

        # select only current event
        mask = mask & (self.track_seg['eventID'] == self._events[self._curr_event])

        # select only tracks in current TPC(s)
        tpc_bound = self.tpc_borders[self._curr_tpc]
        tpc_mask = np.zeros_like(mask)
        for i_tpc in range(self._curr_tpc, min(self._curr_tpc + self.tpc_batch_size, self.tpc_borders.shape[0])):
            tpc_mask = tpc_mask | (
                # overlap in X
                ((self.track_seg['x_end'] > tpc_bound[0,0])
                 | (self.track_seg['x_start'] > tpc_bound[0,0]))
                & ((self.track_seg['x_end'] < tpc_bound[0,1])
                   | (self.track_seg['x_start'] < tpc_bound[0,1]))
                # overlap in Y
                & ((self.track_seg['y_end'] > tpc_bound[1,0])
                   | (self.track_seg['y_start'] > tpc_bound[1,0]))
                & ((self.track_seg['y_end'] < tpc_bound[1,1])
                   | (self.track_seg['y_start'] < tpc_bound[1,1]))
                # overlap in Z
                & ((self.track_seg['z_end'] > tpc_bound[2,0])
                   | (self.track_seg['z_start'] > tpc_bound[2,0]))
                & ((self.track_seg['z_end'] < tpc_bound[2,1])
                   | (self.track_seg['z_start'] < tpc_bound[2,1])))

        self._curr_tpc += self.tpc_batch_size
        mask = mask & tpc_mask
        self._simulated = self._simulated | mask

        if np.any(mask):
            return mask
        else:
            return next(self)

        
        
