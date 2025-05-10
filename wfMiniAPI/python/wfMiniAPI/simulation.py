import json
from wfMiniAPI.component import Component
import time
from wfMiniAPI.kernel import get_kernel_from_string

class Simulation(Component):
    def __init__(self,name="SIM",comm=None,config:dict={"type":"filesystem"}):
        super().__init__(name,config=config)
        self.name = name
        self.comm = comm
        self.kernels = []
        self.ktoi = {}
        if self.comm is not None:
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        else:
            self.size = 1
            self.rank = 0

    def init_from_dict(self, config:dict):
        kernels = config.get('kernels', [])
        for kernel in kernels:
            name = kernel.get('name')
            mini_app_kernel = kernel.get('mini_app_kernel', 'MatMulSimple2D')
            run_count = kernel.get('run_count', 1)
            data_size = tuple(kernel.get('data_size', [32, 32, 32]))
            device = kernel.get('device', 'cpu')
            self.add_kernel(name, mini_app_kernel=mini_app_kernel, device=device, data_size=data_size, run_count=run_count)
            if "run_time" in kernel:
                self.set_kernel_run_count_by_time(name, kernel["run_time"])
    
    def init_from_json(self, json_file):
        """Initialize the simulation from a JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.init_from_dict(data)

    def add_kernel(self, name:str, mini_app_kernel:str="MatMulSimple2D", device:str="cpu", data_size:tuple=(32,32,32), run_count:int=1):
        """Add a kernel to the simulation."""
        kernel_func = get_kernel_from_string(mini_app_kernel)
        self.kernels.append({
            'name': name,
            'func': kernel_func,
            'run_count': run_count,
            'data_size': data_size,
            'device': device
        })
        self.ktoi[name] = len(self.kernels) - 1

    def remove_kernel(self, name):
        """Remove a kernel by name."""
        self.kernels = [k for k in self.kernels if k['name'] != name]

    def set_kernel_run_count(self, name, run_count):
        """Set how many times to run a kernel."""
        for k in self.kernels:
            if k['name'] == name:
                k['run_count'] = run_count

    def set_kernel_data_size(self, name, data_size):
        """Set the data size for a kernel."""
        for k in self.kernels:
            if k['name'] == name:
                k['data_size'] = data_size

    def run(self, nsteps:int=1):
        """Run all kernels in sequence for the specified total_time."""
        self.logger.info(f"Starting simulation {self.name} for {nsteps} steps.")
        for _ in range(nsteps):
            for k in self.kernels:
                for _ in range(k['run_count']):
                    if k['data_size'] is not None:
                        k['func'](k['device'], k['data_size'])
                    else:
                        k['func'](k['device'])
            if self.comm is not None:
                self.comm.Barrier()
        self.logger.info(f"Simulation {self.name} completed {nsteps} steps.")
    
    def set_kernel_run_count_by_time(self, name, total_time):
        """
        Set the run_count for a kernel so that its total execution time is close to total_time.
        Measures the single_run_time automatically.
        Uses self.ktoi to get the kernel index.
        """
        if name not in self.ktoi:
            raise ValueError(f"Kernel '{name}' not found in self.ktoi")
        idx = self.ktoi[name]
        k = self.kernels[idx]
        timing_iter = 100
        # Measure single run time
        if k['data_size'] is not None:
            start = time.time()
            for _ in range(timing_iter):
                k['func'](k['device'], k['data_size'])
            end = time.time()
        else:
            start = time.time()
            for _ in range(timing_iter):
                k['func'](k['device'])
            end = time.time()
        single_run_time = end - start
        if single_run_time <= 0:
            raise ValueError("Measured single_run_time must be positive")
        if self.comm is not None:
            single_run_time = self.comm.allreduce(single_run_time) / self.size
        run_count = int(total_time // (single_run_time/timing_iter))
        self.logger.info(f"Setting run_count for kernel '{name}' to {run_count} based on total_time {total_time} and single_run_time {single_run_time}")
        k['run_count'] = max(1, run_count)
    
    # def set_kernel_data_size_by_time(self, name, total_time, min_data_size=8*8*8, max_data_size=64*64*64,steps=8*8*8):
    #     """
    #     Set the data_size for a kernel so that its total execution time is close to total_time.
    #     Assumes run_count is already set.
    #     Uses self.ktoi to get the kernel index.
    #     """
    #     if name not in self.ktoi:
    #         raise ValueError(f"Kernel '{name}' not found in self.ktoi")
    #     idx = self.ktoi[name]
    #     k = self.kernels[idx]
    #     run_count = k.get('run_count', 1)
    #     if run_count <= 0:
    #         raise ValueError("run_count must be positive to set data_size by time")

    #     data_size = min_data_size
    #     best_data_size = data_size
    #     min_diff = float('inf')
    #     step_size = max(1, (max_data_size - min_data_size) // steps)
    #     for test_size in range(min_data_size, max_data_size + 1, step_size):
    #         start = time.time()
    #         for _ in range(run_count):
    #             k['func'](test_size)
    #         elapsed = time.time() - start
    #         diff = abs(elapsed - total_time)
    #         if diff < min_diff:
    #             min_diff = diff
    #         best_data_size = test_size
    #         if elapsed >= total_time:
    #             break
    #     k['data_size'] = best_data_size