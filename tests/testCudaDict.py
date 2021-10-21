import os

from larndsim.cuda_dict import CudaDict


class testCudaDict:
    test_keys = cp.array([
        0, 1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400])
    test_values = cp.array([
        '0', '1', '2', '3', '4', '10', '20', '30', '40', '100', '200', '300',
        '400'])
    test_default = cp.array(['default'])

    test_lookup_keys_avail = cp.array([
        0, 0, 400, 2, 10, 30, 100])
    test_lookup_values_avail = cp.array([
        '0', '0', '400', '2', '10', '30', '100'])
    test_lookup_keys_unavail = cp.array([
        5, 6, 7, 8, 9, 11, 21, 31, 41, 5000])

    @pytest.fixture
    def cuda_dict():
        cd = CudaDict(default=test_default, tpb=256, bpg=256)
        assert len(cd) == 0
        assert not cp.any(test_keys in cd)
        cd[test_keys] = test_values
        return cd

    def test_init(cuda_dict):
        cd = cuda_dict
        cd.bpg = ceil(len(test_keys) / cd.tpb)
        assert cp.all(test_keys in cd)
        assert cp.all(cd[test_keys] == test_values)

        cd.bpg = ceil(len(test_lookup_keys_avail) / cd.tpb)
        assert cp.all(test_lookup_keys_avail in cd)
        assert cp.all(cd[test_lookup_keys_avail] == test_lookup_values_avail)

        cd.bpg = ceil(len(test_lookup_keys_unavail) / cd.tpb)
        assert np.all(cd[test_lookup_keys_unavail] == test_default[0])
        assert not cp.any(test_lookup_keys_unavail in cd)

    def test_read_write(tmpdir, cuda_dict):
        cd = cuda_dict
        filename = os.path.join(tmpdir, 'test_cd.npz')
        CudaDict.save(filename, cd)

        new_cd = CudaDict.load(filename, tpb=cd.tpb, bpg=cd.bpg)
        assert len(new_cd) == len(cd)
        assert cp.all(new_cd.keys() in cd)
        assert cp.all(cd.keys() in new_cd)
        assert cp.all(cd[new_cd.keys()] == new_cd.values())
        assert cp.all(new_cd[cd.keys()] == cd.values())
