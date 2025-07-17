import subprocess
import os
import sys
import time
import argparse

"""
This is an example on how to build a sim-ai motif from Brewer et. al.[1] 
[1] Brewer, Wes, Ana Gainaru, Frédéric Suter, Feiyi Wang, Murali Emani, and Shantenu Jha. 
"AI-Coupled HPC Workflow Applications, Middleware and Performance.” arXiv, June 20, 2024. https://doi.org/10.48550/arXiv.2406.14315.
Specifically, this is an example of motif-2 in the paper, where simulation sends data to AI, AI trains/finetunes and sends data back to simulation.
"""

def main(np:int,
                other_mpi_opts:str,
                nsim_steps:int,
                write_freq:int,
                dt_type:str,
                dt_location:str):
        sim_cmd = f"mpirun -n {np} {other_mpi_opts} python sim_exec.py --nsteps {nsim_steps} --write_freq {write_freq} --location {dt_location} --type {dt_type}"
        ai_cmd = f"mpirun -n {np} {other_mpi_opts} python ai_exec.py --nsteps {nsim_steps//write_freq} --location {dt_location} --type {dt_type}"

        sim_process = subprocess.Popen(sim_cmd, cwd=os.path.dirname(__file__), shell=True)
        ai_process = subprocess.Popen(ai_cmd, cwd=os.path.dirname(__file__), shell=True)

        finished = False
        while sim_process.poll() is None and ai_process.poll() is None:
        time.sleep(1)
        if sim_process.poll() == 0:
        print("Simulation process finished successfully")
        else:
        print("Simulation process failed")
        if ai_process.poll() == 0:
        print("AI process finished successfully")
        else:
        print("AI process failed")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run the simulation and AI processes")
        parser.add_argument("--np", type=int, default=4, help="Number of processes")
        parser.add_argument("--other_mpi_opts", type=str, default="", help="Other MPI options")
        parser.add_argument("--nsim_steps", type=int, default=10, help="Number of simulation steps")
        parser.add_argument("--write_freq", type=int, default=1, help="Write frequency")
        parser.add_argument("--dt_type", type=str, default="filesystem", help="Data transport type")
        parser.add_argument("--dt_location", type=str, default=os.path.join(os.getcwd(), ".tmp"), help="Data transport location")
        args = parser.parse_args()
        if args.dt_type == "filesystem":
        os.makedirs(args.dt_location, exist_ok=True)
        main(np=args.np, other_mpi_opts=args.other_mpi_opts, nsim_steps=args.nsim_steps, write_freq=args.write_freq, dt_type=args.dt_type, dt_location=args.dt_location)
