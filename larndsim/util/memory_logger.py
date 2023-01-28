import pynvml, tracemalloc, time
import numpy as np

class memory_logger():
    '''
    A class to log memory usage over time. 
    Usage: 
        - Call "start" to begin with.
        
        - Call "take_snapshot" to store the current time stamp, cpu and gpu memory info.
          A caller can provide additional data in the form of a list of values to be stored together.
          This data is stored in the log attribute.
          
        - Call "archive" to store the snapshots taken so far. After this call, the log attribute is
          reset to an empty list. To allow access to the archived data, this function must be called
          with a string in the argument, which can be used to retrieve data anytime.
          
        - Call "savez" to store the archived data into a npz file.

        - If constructed with "do_nothing=True", it doesn't do any of above :)

    Attributes:
    -----------
        log - an intermediate data storage saving a list of snapshot (an array of memory usage etc.)
        data - a data archive which stores a log with an associated string key
    
    Functions:
    -----------
        reset_log()
            - reset the internal state and remove intermediate data stored locally.
        
        start()
            - start the clock, ready to take a snapshot of memory usage
        
        take_snapshot(additional_data)
            - record the time and cpu/gpu memory info in the local intermediate storage.
            ARGS:
                optionally provide a list of floating point values to be stored in addition.
                
        archive(data_name,field_names)
            - store the intermediate data with a key (data_name)
            ARGS:
                data_name
                    - the key string to be associated with the subject data.
                field_names
                    - a list of strings to name variables additionally stored in take_snapshot calls   
        
        savez(filename)
            - store the archived data into a numpy file using numpy.savez function
            ARGS:
                filename
                    - string value to name a npz file
    '''
    lock=None
    
    def __init__(self, do_nothing=False):
        if self.__class__.lock:
            raise RuntimeError(f'Only one instance is allowed for {self.__class__.__name__}')
        self._do_nothing = bool(do_nothing)
        if self._do_nothing:
            return
        pynvml.nvmlInit()
        tracemalloc.start()
        self.log=[]
        self.data=dict()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._t0=None
        self.__class__.lock = True
        
    def __del__(self):
        self.__class__.lock = False
        pynvml.nvmlShutdown()

    def reset_log(self):
        if self._do_nothing: return
        self._t0 = None
        self.log = []
        
    def start(self):
        if self._do_nothing: return
        self.reset_log()
        self._t0 = time.time()
        
    def savez(self,filename='memory_logger.npz'):
        if self._do_nothing: return
        np.savez(filename,**self.data)
        self.data=dict()
        self.reset_log()

    def take_snapshot(self,more_data=[]):
        if self._do_nothing: return
        mem_info = self.catch_memory()
        if len(more_data):
            mem_info = [*mem_info, *more_data]

        # ensure the data size is consistent
        if len(self.log) == 0 or len(self.log[-1]) == len(mem_info):
            self.log.append(mem_info)
        else:
            raise RuntimeError(f'snapshot size is inconsistent (previous {len(self.log[-1])}, current {len(mem_info)})')
    
    def archive(self,data_name,field_names=[]):
        if self._do_nothing: return
        if len(self.log) == 0:
            print('No data logged. Nothing to arxiv...')
            return False
        if data_name in self.data:
            raise KeyError(f'Data name {data_name} already exists')
            
        fields = ['time','cpu_mem_used','cpu_mem_peak','gpu_mem_used','gpu_mem_free']
        default_size = len(fields)
        fields = [*fields,*field_names]
        # check the field length
        if len(fields) > len(self.log[-1]):
            raise RuntimeError('The length of data larger than the names of fields')
            
        if len(fields) < len(self.log[-1]):
            for i in range(len(self.log[-1])):
                if i < default_size:
                    continue
                fields.append(f'data{i-default_size}')
        
        self.data[data_name]   = np.array(self.log,dtype=np.float64)
        self.data[data_name + '_fields'] = fields
        self.reset_log()
        return True
    
    def catch_memory(self):
        if self._do_nothing: return
        if self._t0 is None:
            raise RuntimeError('Must call start function before catch_memory!')
            
        t = time.time() - self._t0
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self._h)
        #mem = psutil.virtual_memory()
        cpu_used, cpu_peak = tracemalloc.get_traced_memory()

        return [t, cpu_used, cpu_peak, gpu_info.used, gpu_info.free]