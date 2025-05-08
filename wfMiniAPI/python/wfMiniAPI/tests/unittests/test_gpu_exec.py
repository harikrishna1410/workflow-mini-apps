from wfMiniAPI.kernel import *
import subprocess


def test_all_compute_kernels():
    executed = False
    kernels = [matMulSimple2D, matMulGeneral, fft, axpy, implaceCompute, generateRandomNumber, scatterAdd]
    if DPNP_AVAILABLE:
        device="xpu"
        for kernel in kernels:
            kernel(device)
        executed = True
    else:
        device="gpu"
        for kernel in kernels:
            kernel(device)
        executed = True
    assert executed, "Not all compute kernels executed successfully."

def test_copy():
    kernels = [dataCopyH2D, dataCopyD2H]
    for kernel in kernels:
        kernel()

def test_mpi_global():
    if DPNP_AVAILABLE:
        device="xpu"
    else:   
        device="gpu"

    cmd = "mpirun -n 4 python3 -c " + f"'from wfMiniAPI.kernel import MPIallReduce; MPIallReduce(\"{device}\")'"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    assert result.returncode == 0, f"MPIallReduce failed to execute. Error: {result.stderr}"
    cmd = "mpirun -n 4 python3 -c " + f"'from wfMiniAPI.kernel import MPIallGather; MPIallGather(\"{device}\")'"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    assert result.returncode == 0, f"MPIallGather failed to execute. Error: {result.stderr}"

if __name__ == "__main__":
    test_all_compute_kernels()
    test_copy()
    test_mpi_global() 
    print("All compute kernels executed successfully.")