import json
from .component import Component
import time

class Simulation(Component):
    def __init__(self,name="SIM"):
        super().__init__(name)
        self.name = name
        self.kernels = []
        self.ktoi = {}

    def add_kernel(self, name, kernel_func, run_count=1, data_size=None):
        """Add a kernel to the simulation."""
        self.kernels.append({
            'name': name,
            'func': kernel_func,
            'run_count': run_count,
            'data_size': data_size
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

    def run(self):
        """Run all kernels in sequence for the specified total_time."""
        for k in self.kernels:
            for _ in range(k['run_count']):
                if k['data_size'] is not None:
                    k['func'](k['data_size'])
                else:
                    k['func']()
    
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
        # Measure single run time
        if k['data_size'] is not None:
            start = time.time()
            k['func'](k['data_size'])
            end = time.time()
        else:
            start = time.time()
            k['func']()
            end = time.time()
        single_run_time = end - start
        if single_run_time <= 0:
            raise ValueError("Measured single_run_time must be positive")
        run_count = int(total_time // single_run_time)
        k['run_count'] = max(1, run_count)
    
    def set_kernel_data_size_by_time(self, name, total_time, min_data_size=8*8*8, max_data_size=64*64*64,steps=8*8*8):
        """
        Set the data_size for a kernel so that its total execution time is close to total_time.
        Assumes run_count is already set.
        Uses self.ktoi to get the kernel index.
        """
        if name not in self.ktoi:
            raise ValueError(f"Kernel '{name}' not found in self.ktoi")
        idx = self.ktoi[name]
        k = self.kernels[idx]
        run_count = k.get('run_count', 1)
        if run_count <= 0:
            raise ValueError("run_count must be positive to set data_size by time")

        data_size = min_data_size
        best_data_size = data_size
        min_diff = float('inf')
        step_size = max(1, (max_data_size - min_data_size) // steps)
        for test_size in range(min_data_size, max_data_size + 1, step_size):
            start = time.time()
            for _ in range(run_count):
                k['func'](test_size)
            elapsed = time.time() - start
            diff = abs(elapsed - total_time)
            if diff < min_diff:
                min_diff = diff
            best_data_size = test_size
            if elapsed >= total_time:
                break
        k['data_size'] = best_data_size