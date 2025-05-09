
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
from mpi4py import MPI
from wfMiniAPI.training import AI
import time
import argparse


def main(n_steps:int, config:dict):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ###NOTE: this not ddp
    ai = AI(f"AI_{rank}", config=config)
    for i in range(n_steps):
        ##comsume the simulation data
        ## wait for the data to be staged
        while not ai.poll_staged_data(f"sim_data_{rank}_{i}"):
            time.sleep(1)
        data = ai.stage_read(f"sim_data_{rank}_{i}")
        ai.train()
        ai.clean_staged_data(f"sim_data_{rank}_{i}")
        ai.infer()
        ai.stage_write(f"ai_data_{rank}_{i}", data)
        comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--nsteps", type=int, required=True, help="Number of training steps")
    parser.add_argument("--location", type=str, required=True, help="Location for the data")
    parser.add_argument("--type", type=str, required=True, help="Transport type for the data")
    args = parser.parse_args()
    config = {"type": args.type, "location": args.location}
    main(args.nsteps, config)
