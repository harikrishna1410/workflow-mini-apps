import numpy as np
import time
import os
import sys

try:
    import cupy as cp
    from cupy.cuda import nccl
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import dpnp as dnp
    DPNP_AVAILABLE = True
except ImportError:
    DPNP_AVAILABLE = False

try:
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True
except ImportError:
    MPI4PY_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


#################
#misc 
#################

def sleep(seconds):
    time.sleep(seconds)

def get_device_module(device):
    if device == "cuda":
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install CuPy to use cuda capabilities.")
        return cp
    elif device == "xpu":
        if not DPNP_AVAILABLE:
            raise ImportError("DPNP not installed")
        return dnp
    else:
        return np

def init_mpi():
    if not MPI.Is_initialized():
        MPI.Init()
    
    
#################
#io
#################

def writeSingleRank(num_bytes, data_root_dir):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            filename = os.path.join(data_root_dir, "data.h5")
            
            num_elem = num_bytes // 4
            data = np.empty(num_elem, dtype=np.float32)
    
            with h5py.File(filename, 'w') as f:
                dset = f.create_dataset("data", data = data)


def writeNonMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(rank))
        else:
            filename = os.path.join(data_root_dir, "data_{}_{}.h5".format(rank, filename_suffix))
        print("In writeNonMPI, rank = ", rank, " filename = ", filename)
        
        num_elem = num_bytes // 4
        data = np.empty(num_elem, dtype=np.float32)

        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("data", data = data)

def writeWithMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        num_elem = num_bytes // 4
        num_elem_tot = num_elem * size
        data = np.empty(num_elem, dtype=np.float32)

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, 'data.h5')
        else:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(filename_suffix))
        print("In writeWithMPI, rank = ", rank, " filename = ", filename)

        with h5py.File(filename, 'w', driver='mpio', comm=comm) as f:
            dset = f.create_dataset("data", (num_elem_tot, ), dtype=np.float32)
            offset = rank * num_elem
            dset[offset:offset+num_elem] = data

def readNonMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(rank))
        else:
            filename = os.path.join(data_root_dir, "data_{}_{}.h5".format(rank, filename_suffix))
        print("In readNonMPI, rank = ", rank, " filename = ", filename)
        
        num_elem = num_bytes // 4

        with h5py.File(filename, 'r') as f:
            data = f['data'][0:num_elem] 

def readWithMPI(num_bytes, data_root_dir, filename_suffix=None):
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to use multi-process read/write.")
    elif not H5PY_AVAILABLE:
        raise ImportError("h5py is not installed. Install h5py to use read/write.")
    else:
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        num_elem = num_bytes // 4
        num_elem_tot = num_elem * size
        data = np.empty(num_elem, dtype=np.float32)

        if filename_suffix == None:
            filename = os.path.join(data_root_dir, 'data.h5')
        else:
            filename = os.path.join(data_root_dir, "data_{}.h5".format(filename_suffix))
        print("In readWithMPI, rank = ", rank, " filename = ", filename)

        with h5py.File(filename, 'r', driver='mpio', comm=comm) as f:
            dset = f['data']
            offset = rank * num_elem
            dset.read_direct(data, np.s_[offset:offset+num_elem])


#################
#comm 
#################

def MPIallReduce(device:str, data_size:tuple=(32,32), backend:str="mpi"):
    xp = get_device_module(device)
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to perform allreduce.")
    else:
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        sendbuf = xp.empty(data_size, dtype=xp.float32)
        recvbuf = xp.empty(data_size, dtype=xp.float32)
        
        if device == "cpu":
            comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
    
        elif device == "cuda":
            if backend == "nccl":
                uid = nccl.get_unique_id()
                comm_nccl = nccl.NcclCommunicator(size, uid, rank)
                comm_nccl.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, data_size, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, cp.cuda.Stream.null)
                cp.cuda.Stream.null.synchronize()
            elif backend == "mpi":
                sendbuf_h = cp.asnumpy(sendbuf)
                recvbuf_h = cp.asnumpy(recvbuf)
                comm.Allreduce(sendbuf_h, recvbuf_h, op=MPI.SUM)
            else:
                raise ValueError(f"Invalid backend {backend}. Choose either 'nccl' or 'mpi'.")
        elif device == "xpu":
            if backend == "mpi":
                sendbuf_h = dnp.asnumpy(sendbuf)
                recvbuf_h = dnp.asnumpy(recvbuf)
                comm.Allreduce(sendbuf_h, recvbuf_h, op=MPI.SUM)
            else:
                raise ValueError(f"Invalid backend {backend}. Choose 'mpi'.")
        else:
            raise ValueError(f"Invalid device {device}. Choose either 'cpu', 'cuda', or 'xpu'.")
    
def MPIallGather(device:str, data_size:tuple=(32,32), backend:str="mpi"):
    xp = get_device_module(device)
    if not MPI4PY_AVAILABLE:
        raise ImportError("mpi4py is not installed. Install mpi4py to perform allgather.")
    else:
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialized. Please initialize MPI before calling this function.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        sendbuf = xp.empty(data_size, dtype=xp.float32)
        recvbuf = xp.empty(tuple([data_size[i] for i in range(len(data_size)-1)] + [data_size[-1] * size]), dtype=xp.float32)

        if device == "cpu":
            comm.Allgather(sendbuf, recvbuf)

        elif device == "cuda":
            if backend == "mpi":
                sendbuf_h = cp.asnumpy(sendbuf)
                recvbuf_h = cp.asnumpy(recvbuf)
                comm.Allgather(sendbuf_h, recvbuf_h)
            elif backend == "nccl":
                uid = nccl.get_unique_id()
                comm_nccl = nccl.NcclCommunicator(size, uid, rank)
                comm_nccl.allGather(sendbuf.data.ptr, recvbuf.data.ptr, data_size, nccl.NCCL_FLOAT32, cp.cuda.Stream.null)
                cp.cuda.Stream.null.synchronize()
            else:
                raise ValueError(f"Invalid backend {backend}. Choose either 'nccl' or 'mpi'.")
        elif device == "xpu":
            if backend == "mpi":
                sendbuf_h = dnp.asnumpy(sendbuf)
                recvbuf_h = dnp.asnumpy(recvbuf)
                comm.Allgather(sendbuf_h, recvbuf_h)
            else:
                raise ValueError(f"Invalid backend {backend}. Choose 'mpi'.")
        else:   
            raise ValueError(f"Invalid device {device}. Choose either 'cpu', 'cuda', or 'xpu'.")


#################
#data movement
#################

def dataCopyH2D(data_size:tuple=(32,32,32)):
    if not CUPY_AVAILABLE and not DPNP_AVAILABLE:
        raise ImportError("CuPy or DPNP is not installed.")

    # Allocate array on the host (CPU)
    # Then transfer to the selected device
    data_h = np.empty(data_size, dtype=np.float32)
    if DPNP_AVAILABLE:
        data_d = dnp.array(data_h)
    else:
        data_d = cp.asarray(data_h)

def dataCopyD2H(data_size:tuple=(32,32,32)):
    if not CUPY_AVAILABLE and not DPNP_AVAILABLE:
        raise ImportError("CuPy or DPNP is not installed.")

    if DPNP_AVAILABLE:
        data_d = dnp.empty(data_size, dtype=dnp.float32)
        data_h = dnp.asnumpy(data_d)
    else:
        data_d = cp.empty(data_size, dtype=cp.float32)
        data_h = cp.asnumpy(data_d)


#################
#computation
#################

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union, Any

class ComputeKernel(ABC):
    """ This ia base class for all compute kernels."""
    @abstractmethod
    def __call__(self, device:str, data_size:tuple=(32,32,32),**kwargs) -> Any:
        """
        This method should be implemented by subclasses to define the kernel's behavior.
        :param device: The device to use for computation (e.g., 'cpu', 'cuda', 'xpu').
        :param data_size: The size of the data to be processed.
        :param kwargs: Additional arguments for the kernel.
        :return: The result of the computation.
        """
        pass

class MatMulSimple2D(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        xp = get_device_module(device)
        matrix_a = xp.empty(data_size, dtype=xp.float32)
        matrix_b = xp.empty(data_size, dtype=xp.float32)
        return xp.matmul(matrix_a, matrix_b)

class MatMulGeneral(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), axis: int | tuple = 2, **kwargs):
        xp = get_device_module(device)
        matrix_a = xp.empty(data_size, dtype=xp.float32)
        matrix_b = xp.empty(data_size, dtype=xp.float32)
        return xp.tensordot(matrix_a, matrix_b, axis)

class FFT(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), type_in: str = "float", transform_dim: int = -1, **kwargs):
        xp = get_device_module(device)
        if type_in == "float":
            data_in = xp.empty(data_size, dtype=xp.float32)
        elif type_in == "double":
            data_in = xp.empty(data_size, dtype=xp.float64)
        elif type_in == "complexF":
            data_in = xp.empty(data_size, dtype=xp.complex64)
        elif type_in == "complexD":
            data_in = xp.empty(data_size, dtype=xp.complex128)
        else:
            raise TypeError("In fft call, type_in must be one of the following: [float, double, complexF, complexD]")
        
        return xp.fft.fft(data_in, axis=transform_dim)

class AXPY(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        xp = get_device_module(device)
        x = xp.empty(data_size, dtype=xp.float32)
        y = xp.empty(data_size, dtype=xp.float32)
        y += 1.01 * x
        return y

class InplaceCompute(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), op="exp", **kwargs):
        xp = get_device_module(device)
        x = xp.empty(data_size, dtype=xp.float32)
        # op can be either a string identifier or a Python callable
        if isinstance(op, str):
            if op == "exp":
                op_func = xp.exp
            else:
                raise ValueError(f"Unknown operator {op}.")
        else:
            if not callable(op):
                raise ValueError("Operator must be a callable function.")
            else:
                op_func = op

        return op_func(x)

class GenerateRandomNumber(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        xp = get_device_module(device)
        return xp.random.rand(*data_size)

class ScatterAdd(ComputeKernel):
    def __call__(self, device: str, data_size: tuple = (32, 32, 32), **kwargs):
        xp = get_device_module(device)
        y = xp.empty(xp.prod(data_size), dtype=xp.float32)
        x = xp.empty(xp.prod(data_size), dtype=xp.float32)
        idx = xp.random.randint(0, xp.prod(data_size), size=xp.prod(data_size), dtype=xp.int32)
        if device.lower() == "cpu":
            y += x[idx]
        elif device.lower() == "cuda":
            scatter_add_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void my_scatter_add_kernel(const float *x, const float *y, const int *idx)
            {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                // Implementation needed
            }
            ''', 'my_scatter_add_kernel')
            # Implementation needed
        return y

def get_kernel_from_string(kernel_name: str):
    kernel_name_lower = kernel_name.lower()
    # Loop through all compute kernel subclasses
    for kernel_class in ComputeKernel.__subclasses__():
        if kernel_name_lower == kernel_class.__name__.lower():
            return kernel_class()
    
    # If no match found
    raise ValueError(f"Unknown kernel name {kernel_name}.")

def get_all_compute_kernels():
    """
    Returns a list of all compute kernels.
    """
    return [kernel_class() for kernel_class in ComputeKernel.__subclasses__()]

