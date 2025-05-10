from wfMiniAPI.simulation import Simulation
import os
import subprocess


def test_simulation_run():
    sim = Simulation()
    sim.add_kernel("MatMulSimple2D")
    sim.add_kernel("MatMulGeneral")
    sim.run(nsteps=10)

def test_run_mpi():
    cmd  = ["mpiexec", "-n", "2", "python3", "-c", 
            "'from mpi4py import MPI; from wfMiniAPI.simulation import Simulation; comm = MPI.COMM_WORLD; sim = Simulation(comm); sim.add_kernel(\"MatMulSimple2D\"); sim.run(nsteps=10)'"]
    result = subprocess.run(cmd, shell=True, capture_output=True)
    assert result.returncode == 0, f"Error running MPI simulation: {result.stderr.decode()}"

def test_simulation_init_from_json():
    sim = Simulation()
    sim.init_from_json(os.path.join(os.path.dirname(__file__), "init_sim.json"))
    sim.run(nsteps=10)

def test_set_run_count():
    sim = Simulation()
    sim.add_kernel("MatMulSimple2D")
    sim.set_kernel_run_count_by_time("MatMulSimple2D", 1.0)
    sim.run(nsteps=10)


if __name__ == "__main__":
    test_simulation_run()
    test_simulation_init_from_json()
    test_set_run_count()
    test_run_mpi()
    print("All tests passed!")