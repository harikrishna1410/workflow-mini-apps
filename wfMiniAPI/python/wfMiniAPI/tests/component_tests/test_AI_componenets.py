from wfMiniAPI.training import AI
import subprocess


def test_ai_cpu():
    ai_component = AI(num_hidden_layers=2)
    ai_component.train()
    ai_component.infer()
    return

def test_ai_gpu():
    try:
        ai_component = AI(num_hidden_layers=2,device="cuda")
        ai_component.train()
        ai_component.infer()
    except:
        ai_component = AI(num_hidden_layers=2,device="xpu")
        ai_component.train()
        ai_component.infer()
    return

def test_ai_ddp():
    cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI;from wfMiniAPI.training import AI;comm = MPI.COMM_WORLD;AI(num_hidden_layers=2,ddp=True,comm=comm).train()'"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    assert result.returncode == 0, f"Distributed AI component failed to execute. Error: {result.stderr}"

    try:
        cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI;from wfMiniAPI.training import AI;comm = MPI.COMM_WORLD;AI(num_hidden_layers=2,ddp=True,device=\"cuda\",comm=comm).train()'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        assert result.returncode == 0, f"Distributed AI component failed to execute. Error: {result.stderr}"
    except:
        cmd = "mpirun -n 4 python3 -c " + f"'from mpi4py import MPI;from wfMiniAPI.training import AI;comm = MPI.COMM_WORLD;AI(num_hidden_layers=2,ddp=True,device=\"xpu\",comm=comm).train()'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        assert result.returncode == 0, f"Distributed AI component failed to execute. Error: {result.stderr}"

    return

if __name__ == "__main__":
    test_ai_cpu()
    test_ai_gpu()
    test_ai_ddp()
    print("AI component tests passed.")