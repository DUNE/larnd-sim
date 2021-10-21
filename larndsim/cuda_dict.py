from numba import cuda
import cupy as cp


_EMPTY_KEY = 0xFFFFFFFF


class CudaDict(object):
    '''
        A numba implementation of a static hash table that lives on the GPU.
        Based on this project
        https://github.com/nosferalatu/SimpleGPUHashTable.git

        Initialization is preformed via::

            cd = CudaDict(default=cp.array([0.]), tpb=256, bpg=256)
            cd[keys] = values

        Lookup is available via::

            values = cd[keys]

        ``keys`` are expected to be a 1D array of integer key values. When
        initializing, ``keys`` must be unique.

        Removal of items can be performed via::

            del cd[keys]

    '''

    def __init__(self, default, tpb, bpg):
        self.tpb = tpb
        self.bpg = bpg
        self.default = default
        self._hashtable_keys = cp.full(1, _EMPTY_KEY, dtype=np.int32)
        self._hashtable_values = cp.empty(1, dtype=default.dtype)

    def keys(self):
        mask = self._hashtable_keys != _EMPTY_KEY
        return self._hashtable_keys[mask]

    def values(self):
        mask = self._hashtable_keys != _EMPTY_KEY
        return self._hashtable_values[mask]

    def items(self):
        return self.keys(), self.values()

    def __getitem__(self, key):
        encoding = cp.empty(key.shape[0], dtype=np.int32)
        values = cp.empty(key.shape[0], dtype=self._hashtable_values.dtype)
        cuda_hashtable_lookup[self.tpb, self.bpg](
            key, values, encoding, self._hashtable_keys, self._hashtable_values,
            self.default)
        return values

    def __setitem__(self, key, value):
        if len(self) == 0:
            self._hashtable_keys = cp.full(key.shape[0] + 1, _EMPTY_KEY, dtype=np.int32)
            self._hashtable_values = cp.empty(key.shape[0] + 1, dtype=value.dtype)
        else:
            raise NotImplementedError('Trying to update CudaDict, not yet supported')
        encoding = cp.empty(key.shape[0], dtype=np.int32)
        cuda_hashtable_insert[self.tpb, self.bpg](
            key, value, encoding, self._hashtable_keys, self._hashtable_values)

    def __delitem__(self, key):
        if len(self) == 0:
            return
        encoding = cp.empty(key.shape[0], dtype=np.int32)
        cuda_hashtable_delete[self.tpb, self.bpg](
            key, encoding, self._hashtable_key_arr, self._hashtable_value_arr)

    def __len__(self):
        return len(self._hashtable_keys) - 1

    def __contains__(self, key):
        exists = cp.zeros(key.shape[0], dtype=bool)
        if len(self) == 0:
            return exists
        encoding = cp.empty(key.shape[0], dtype=np.int32)
        cuda_hashtable_exists[self.tpb, self.bpg](
            key, exists, encoding, self._hashtable_keys)
        return exists

    @staticmethod
    def load(filename, tpb, bpg):
        data = cp.load(filename)
        keys = data['keys']
        values = data['values']
        default = data['default']
        cd = CudaDict(default=default, tpb=tpb, bpg=bpg)
        cd[keys] = values
        return cd

    @staticmethod
    def save(filename, cdict):
        mask = cdict._hashtable_keys != _EMPTY_KEY
        keys = cdict._hashtable_keys[mask]
        values = cdict._hashtable_values[mask]
        default = cdict.default
        data = dict(
            keys=keys,
            values=values,
            default=default
        )
        cp.savez_compressed(filename, **data)


_HASH_CONSTANT = 257


@cuda.jit
def cuda_hashtable_encode(key_arr, entries, encoding_arr):
    '''
    Encodes keys into a integer number between 0 and entries-1
    Args:
        key_arr (: obj: `numpy.ndarray`): list of keys
        entries (int): maximum integer value to return
        encoding_arr (: obj: `numpy.ndarray`): output array
    '''
    ikey = cuda.grid(1)

    if ikey < key_arr.shape[0]:
        encoding_arr[ikey] = ((key_arr[ikey] * _HASH_CONSTANT) % entries)


@cuda.jit
def cuda_hashtable_insert(key_arr, value_arr, encoding_arr, hashtable_key_arr, hashtable_value_arr):
    '''
    Inserts keys into a hashtable
    Args:
        key_arr (: obj: `numpy.ndarray`): list of keys to insert
        value_arr (: obj: `numpy.ndarray`): list of values to insert
        encoding_arr (: obj: `numpy.ndarray`): temporary array used to store hash indices
        hashtable_key_arr (: obj: `numpy.ndarray`): list of unique keys in hashtable
        hashtable_value_arr (: obj: `numpy.ndarray`): list of values in hashtable
    '''
    ikey = cuda.grid(1)

    if ikey < key_arr.shape[0]:

        cuda_hashtable_encode(key_arr[ikey], hashtable_key_arr.shape[0], encoding_arr[ikey])
        while True:
            cuda.atomic.compare_and_swap(hashtable_key_arr[encoding_arr[ikey]], _EMPTY_KEY, key_arr[ikey])
            if hashtable_key_arr[encoding_arr[ikey]] == key_arr[ikey]:
                hashtable_value_arr[encoding_arr[ikey]] = value_arr[ikey]
                break
            encoding_arr[ikey] += (encoding_arr[ikey] + 1) % hashtable_key_arr.shape[0]


@cuda.jit
def cuda_hashtable_lookup(key_arr, value_arr, encoding_arr, hashtable_key_arr,
                          hashtable_value_arr, default):
    '''
    Fetches values from hashtable
    Args:
        key_arr (: obj: `numpy.ndarray`): list of keys to lookup
        value_arr (: obj: `numpy.ndarray`): output array
        encoding_arr (: obj: `numpy.ndarray`): temporary array used to store hash indices
        hashtable_key_arr (: obj: `numpy.ndarray`): list of unique keys in hashtable
        hashtable_value_arr (: obj: `numpy.ndarray`): list of values in hashtable
        default (: obj: `numpy.ndarray`): singleton array to use if key is not present
    '''
    ikey = cuda.grid(1)

    if ikey < key_arr.shape[0]:

        cuda_hashtable_encode(key_arr[ikey], hashtable_key_arr.shape[0], encoding_arr[ikey])
        while True:
            if hashtable_key_arr[encoding_arr[ikey]] == key_arr[ikey]:
                value_arr[ikey] = hashtable_value_arr[encoding_arr[ikey]]
                break
            if hashtable_key_arr[encoding_arr[ikey]] == _EMPTY_KEY:
                value_arr[ikey] = default[0]
                break
            encoding_arr[ikey] += (encoding_arr[ikey] + 1) % hashtable_key_arr.shape[0]


@cuda.jit
def cuda_hashtable_exists(key_arr, exists_arr, encoding_arr, hashtable_key_arr):
    '''
    Checks if a key is present in the hashtable
    Args:
        key_arr (: obj: `numpy.ndarray`): list of keys to remove
        exists_arr (: obj: `numpy.ndarray`): output array
        encoding_arr (: obj: `numpy.ndarray`): temporary array used to store hash indices
        hashtable_key_arr (: obj: `numpy.ndarray`): list of unique keys in hashtable
    '''
    ikey = cuda.grid(1)

    if ikey < key_arr.shape[0]:

        cuda_hashtable_encode(key_arr[ikey], hashtable_key_arr.shape[0], encoding_arr[ikey])
        while True:
            if hashtable_key_arr[encoding_arr[ikey]] == key_arr[ikey]:
                exists_arr[ikey] = True
                break
            if hashtable_key_arr[encoding_arr[ikey]] == _EMPTY_KEY:
                exists_arr[ikey] = False
                break
            encoding_arr[ikey] += (encoding_arr[ikey] + 1) % hashtable_key_arr.shape[0]


@cuda.jit
def cuda_hashtable_delete(key_arr, encoding_arr, hashtable_key_arr, hashtable_value_arr):
    '''
    Removes a key from the hashtable
    Args:
        key_arr (: obj: `numpy.ndarray`): list of keys to remove
        encoding_arr (: obj: `numpy.ndarray`): temporary array used to store hash indices
        hashtable_key_arr (: obj: `numpy.ndarray`): list of unique keys in hashtable
        hashtable_value_arr (: obj: `numpy.ndarray`): list of values in hashtable
    '''
    ikey = cuda.grid(1)

    if ikey < key_arr.shape[0]:

        cuda_hashtable_encode(key_arr[ikey], hashtable_key_arr.shape[0], encoding_arr[ikey])
        while True:
            if hashtable_key_arr[encoding_arr[ikey]] == key_arr[ikey]:
                hashtable_key_arr[encoding_arr[ikey]] == _EMPTY_KEY
                break
            if hashtable_key_arr[encoding_arr[ikey]] == _EMPTY_KEY:
                break
            encoding_arr[ikey] += (encoding_arr[ikey] + 1) % hashtable_key_arr.shape[0]
