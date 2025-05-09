from wfMiniAPI.kernel import *
import subprocess


def test_all_compute_kernels():
    kernels = get_all_compute_kernels()
    device = "cpu"
    for kernel in kernels:
        kernel(device)
    
    if DPNP_AVAILABLE:
        device="xpu"
        for kernel in kernels:
            kernel(device)
        executed = True
    elif CUPY_AVAILABLE:
        device="cuda"
        for kernel in kernels:
            kernel(device)
        executed = True
    else:
        print("No GPU support available. Skipping GPU tests.")
    

def test_copy():
    if not DPNP_AVAILABLE and not CUPY_AVAILABLE:
        print("No GPU support available. Skipping copy tests.")
        return
    kernels = [dataCopyH2D, dataCopyD2H]
    for kernel in kernels:
        kernel()

def test_mpi_global():
    def check_mpi(device):
        cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI; from wfMiniAPI.kernel import MPIallReduce; MPIallReduce(\"{device}\")'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        assert result.returncode == 0, f"MPIallReduce failed to execute. Error: {result.stderr}"
        cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI; from wfMiniAPI.kernel import MPIallGather; MPIallGather(\"{device}\")'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        assert result.returncode == 0, f"MPIallGather failed to execute. Error: {result.stderr}"

    check_mpi("cpu")
    if DPNP_AVAILABLE:
        check_mpi("xpu")
    elif CUPY_AVAILABLE:
        check_mpi("cuda")
    else:
        print("No GPU support available. Skipping GPU-MPI tests.")
        return

if __name__ == "__main__":
    test_all_compute_kernels()
    test_copy()
    test_mpi_global()
    print("All compute kernels executed successfully.")