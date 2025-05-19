import json
from wfMiniAPI.component import Component
import time
from wfMiniAPI.kernel import *
import os
import sys
import logging as _logging

class Simulation(Component):
    def __init__(self,name="SIM",comm=None,config:dict={"type":"filesystem"},logging=False,log_level=_logging.INFO):
        super().__init__(name,config=config,logging=logging,log_level=log_level)
        self.name = name
        self.comm = comm
        self.kernels = []
        self.ktoi = {}
        if self.comm is not None:
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            self.local_rank = os.environ.get("MPI_LOCALRANKID",0)
            if self.logger:
                self.logger.debug(f"Setting the local rank as {self.local_rank}")
        else:
            self.size = 1
            self.rank = 0
            self.local_rank = 0

    def init_from_dict(self, config:dict):
        kernels = config.get('kernels', [])
        for kernel in kernels:
            name = kernel.get('name')
            mini_app_kernel = kernel.get('mini_app_kernel', 'MatMulSimple2D')
            assert "run_time" in kernel or "run_count" in kernel, "Kernel should either have run_count or run_time"
            run_time = kernel.get("run_time",None)
            run_count = kernel.get('run_count',None)
            data_size = tuple(kernel.get('data_size', [32, 32, 32]))
            device = kernel.get('device', 'cpu')
            self.add_kernel(name, mini_app_kernel=mini_app_kernel, device=device, data_size=data_size, run_count=run_count, run_time=run_time)

    
    def init_from_json(self, json_file):
        """Initialize the simulation from a JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.init_from_dict(data)

    def add_kernel(self, 
                    name:str, 
                    mini_app_kernel:str="MatMulSimple2D", 
                    device:str="cpu", 
                    data_size:tuple=(32,32,32), 
                    run_count:int=None, 
                    run_time:float=None):
        """Add a kernel to the simulation."""
        kernel_func = get_kernel_from_string(mini_app_kernel)
        assert run_time is not None or run_count is not None
        self.kernels.append({
            'name': name,
            'func': kernel_func,
            'run_count': run_count,
            'run_time': run_time,
            'data_size': data_size,
            'device': device
        })
        self.ktoi[name] = len(self.kernels) - 1

    def remove_kernel(self, name):
        """Remove a kernel by name."""
        self.kernels = [k for k in self.kernels if k['name'] != name]

    def set_kernel_run_count(self, name:str, run_count:int):
        """Set how many times to run a kernel."""
        if name in self.ktoi:
            self.kernels[self.ktoi[name]]['run_count'] = run_count
        else:
            raise ValueError(f"Unknow kernel {name}")
    
    def set_kernel_run_time(self, name:str, run_time:float):
        if name in self.ktoi:
            self.kernels[self.ktoi[name]]['run_time'] = run_time
        else:
            raise ValueError(f"Unknow kernel {name}")

    def set_kernel_data_size(self, name:str, data_size:tuple):
        """Set the data size for a kernel."""
        if name in self.ktoi:
            self.kernels[self.ktoi[name]]['data_size'] = data_size
        else:
            raise ValueError(f"Unknow kernel {name}")

    def run(self, nsteps:int=1) -> float:
        """Run 1 iteration of the simulation"""
        total_dt = 0.0
        for k in self.kernels:
            if k["run_time"] is not None:
                kernel_dt = 0.0
                if isinstance(k['func'],ComputeKernel):
                    while kernel_dt < k["run_time"]:
                        host_dt,device_dt = k['func'](k['device'], k['data_size'])
                        kernel_dt += host_dt
                else:
                    raise ValueError("Sorry, unsupported kernel type :-(")
                
                if self.logger:
                    self.logger.debug(f"Actual runtime {kernel_dt} target runtime {k['run_time']}")
            elif k["run_count"] is not None:
                kernel_dt = 0.0
                if isinstance(k['func'],ComputeKernel):
                    for _ in range(k["run_count"]):
                        host_dt,device_dt = k['func'](k['device'], k['data_size'])
                        kernel_dt += host_dt
                else:
                    raise ValueError("Sorry, unsupported kernel type :-(")
                if self.logger:
                    self.logger.debug(f"Kernel run time {kernel_dt} for {k['run_count']} iterations")
            else:
                raise ValueError(f"kernel {k['name']} doesn't have neither run_count not run_time ")
            total_dt += kernel_dt
        if self.comm is not None:
            ##no all reduce to avoid unecessary communications 
            self.comm.Barrier()
        return total_dt

###This was quite unrelaible. So, switched to keeping count or run_time during the self.run itself     
    # def set_kernel_run_count_by_time(self, name, total_time):
    #     """
    #     Set the run_count for a kernel so that its total execution time is close to total_time.
    #     Measures the single_run_time automatically.
    #     Uses self.ktoi to get the kernel index.
    #     """
    #     if name not in self.ktoi:
    #         raise ValueError(f"Kernel '{name}' not found in self.ktoi")
    #     idx = self.ktoi[name]
    #     k = self.kernels[idx]
    #     ###Simply warmup the gpu.
    #     timing_iter = 100
    #     for _ in range(timing_iter):
    #         host_dt,device_dt = k['func'](k['device'], k['data_size'])
    #     # Measure single run time
    #     total_dt = 0.0
    #     if k['data_size'] is not None:
    #         for i in range(timing_iter):
    #             host_dt,device_dt = k['func'](k['device'], k['data_size'])
    #             total_dt += device_dt
    #     else:
    #         for _ in range(timing_iter):
    #             host_dt,device_dt = k['func'](k['device'])
    #             total_dt += device_dt
    #     single_run_time = total_dt
    #     if single_run_time <= 0:
    #         raise ValueError("Measured single_run_time must be positive")
    #     if self.comm is not None:
    #         single_run_time = self.comm.allreduce(single_run_time) / self.size
    #     run_count = int(total_time // (single_run_time/timing_iter))
    #     assert run_count > 0, f"runcount == 0 try reducing the data size target_time: {total_time}, iteration time {single_run_time/timing_iter} {device_dt} {host_dt}"
    #     if self.logger:
    #         self.logger.info(f"Setting run_count for kernel '{name}' to {run_count} based on total_time {total_time} and single_run_time {single_run_time/timing_iter} {device_dt}")
    #     k['run_count'] = max(1, run_count)
    
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