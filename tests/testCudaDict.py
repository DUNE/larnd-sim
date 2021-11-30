import os
import pytest
import cupy as cp
from math import ceil

from larndsim.cuda_dict import CudaDict


class TestCudaDict:
    test_keys = cp.array([
        0, 1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400])
    test_values = test_keys.astype(float) + 1.
    test_default = cp.array([999.], dtype=float)

    test_lookup_keys_avail = cp.array([
        0, 0, 400, 2, 10, 30, 100])
    test_lookup_values_avail = test_lookup_keys_avail.astype(float) + 1.
    test_lookup_keys_unavail = cp.array([
        5, 6, 7, 8, 9, 11, 21, 31, 41, 5000])

    def init_cuda_dict(self):
        cd = CudaDict(default=self.test_default, tpb=256, bpg=1)
        assert len(cd) == 0
        assert not cp.any(cd.contains(self.test_keys))
        cd[self.test_keys] = self.test_values
        return cd

    def test_init(self):
        cd = self.init_cuda_dict()
        cd.bpg = ceil(len(self.test_keys) / cd.tpb)
        assert cp.all(cd.contains(self.test_keys))
        assert cp.all(cd[self.test_keys] == self.test_values)

        cd.bpg = ceil(len(self.test_lookup_keys_avail) / cd.tpb)
        assert cp.all(cd.contains(self.test_lookup_keys_avail))
        assert cp.all(cd[self.test_lookup_keys_avail] == self.test_lookup_values_avail)

        cd.bpg = ceil(len(self.test_lookup_keys_unavail) / cd.tpb)
        assert cp.all(cd[self.test_lookup_keys_unavail] == self.test_default[0])
        assert not cp.any(cd.contains(self.test_lookup_keys_unavail))

    def test_read_write(self, tmpdir):
        cd = self.init_cuda_dict()
        filename = os.path.join(tmpdir, 'test_cd.npz')
        CudaDict.save(filename, cd)

        new_cd = CudaDict.load(filename, tpb=cd.tpb)
        assert len(new_cd) == len(cd)
        assert cp.all(cd.contains(new_cd.keys()))
        assert cp.all(new_cd.contains(cd.keys()))
        assert cp.all(cd[new_cd.keys()] == new_cd.values())
        assert cp.all(new_cd[cd.keys()] == cd.values())
