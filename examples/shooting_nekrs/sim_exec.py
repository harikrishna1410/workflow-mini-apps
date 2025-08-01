from mpi4py import MPI
from wfMiniAPI.simulation import Simulation as sim
import argparse
import numpy as np
import logging as logging_
import time

"""
Here a simulation is defined using sim_telemetry.json. This telemetry data has the following details  
    "name": name of the simulation step,
    "mini_app_kernel":name of the mini app kernel to be used (optional),
    "run_count": number of times to run the kernel (optional),
    "data_size": size of the data to be used (optional),
    "device": device to be used (optional),
    "run_time": time to run the kernel. This takes precedence over "run_count" (optional),

    the simulation simply runs for certaing numer of steps and stages data for the AI to read.
    Then waits for the AI to consume the data and put its inference result back to the simulation.
"""

def main(write_freq=1,data_size=32*32*32,dtype=np.float64,config={"type":"filesystem"}):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a simulation object
    logging = rank==0
    simulation = sim(name=f"sim_{rank}", comm=comm, config=config,logging=logging)

    # Initialize the simulation from a JSON file
    simulation.init_from_json("sim_telemetry.json")
    comm.Barrier()
    i=0
    while i < 1000:
        tic = time.time()
        # Run the simulation step
        iter_dt_out = simulation.run(nsteps=1)
        iter_time = time.time() - tic
        # Stage data for the AI to read
        if i % write_freq == 0:
            tic = time.time()
            if simulation.logger:
                simulation.logger.debug(f"Write the data: sim_data_{rank}_{i//write_freq}")
            simulation.stage_write(f"sim_data_{rank}_{i//write_freq}", np.empty(data_size, dtype=dtype))
            toc = time.time()
            data_write_time = toc - tic
        else:
            data_write_time = 0.0
        
        if simulation.logger:
            simulation.logger.info(f"tstep time: {iter_time}, returned tstep time {iter_dt_out}, dt time: {data_write_time}")
        ##check for data
        if simulation.poll_staged_data(f"ai_data_{rank}"):
            data = simulation.stage_read(f"ai_data_{rank}")
            if data=="kill_sim":
                if simulation.logger:
                    simulation.logger.info("Received kill message!")
                break
        i+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_freq", type=int, default=1, help="Frequency of writing simulation data")
    parser.add_argument("--data_size", type=int, default=32*32*32, help="Size of the data to stage")
    parser.add_argument("--dtype", type=str, default="float64", help="Data type of the simulation data")
    parser.add_argument("--location", type=str, required=True, help="Location for the simulation data")
    parser.add_argument("--type", type=str, required=True, help="Transport type for the simulation data")
    args = parser.parse_args()
    # Convert dtype string to numpy dtype
    if args.dtype == "float32":
        dtype = np.float32
    elif args.dtype == "float64":
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    config = {"type": args.type, "location": args.location}
    main(write_freq=args.write_freq, data_size=args.data_size, dtype=dtype, config=config)
