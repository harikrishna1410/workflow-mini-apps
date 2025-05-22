from mpi4py import MPI
from wfMiniAPI.simulation import Simulation as sim
import argparse
import numpy as np
import logging as logging_
import time
import json

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

def main(write_freq:int,data_size:int,dtype,config:dict,niters:int,sim_id:int,db_addresses:str=None,ppn:int=None):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a simulation object
    logging = rank==0
    if config["type"] == "redis":
        assert db_addresses is not None and ppn is not None
        config["server-address"] = db_addresses.split(",")[rank//ppn]
    simulation = sim(name=f"sim_{sim_id}", comm=comm, config=config,logging=logging,log_level=logging_.INFO)

    # Initialize the simulation from a JSON file
    simulation.init_from_json("sim_telemetry.json")
    comm.Barrier()
    i=0
    while i < niters:
        tic = time.time()
        # Run the simulation step
        iter_dt_out = simulation.run(nsteps=1)
        iter_time = time.time() - tic
        # Stage data for the AI to read
        if i % write_freq == 0:
            tic = time.time()
            if simulation.logger:
                simulation.logger.debug(f"Write the data: sim_data_{rank}_{i//write_freq}")
            simulation.stage_write(f"sim_{sim_id}_{rank}_{i//write_freq}", np.empty(data_size, dtype=dtype))
            toc = time.time()
            data_write_time = toc - tic
        else:
            data_write_time = 0.0
        
        if simulation.logger:
            simulation.logger.info(f"tstep time: {iter_time}, returned tstep time {iter_dt_out}, dt time: {data_write_time}")
        # ##check for data
        # if simulation.poll_staged_data(f"ai_data_{rank}"):
        #     data = simulation.stage_read(f"ai_data_{rank}")
        #     if data=="kill_sim":
        #         if simulation.logger:
        #             simulation.logger.info("Received kill message!")
        #         break
        i+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True, help="Sim config")
    parser.add_argument("--sim_id", type=int, required=True, help="Sim index")
    parser.add_argument("--db_addresses", type=str, default=None)
    args = parser.parse_args()

    with open(args.config_name,"r") as f:
        config = json.load(f)    

    
    main(config["write_freq"], config["data_size"], np.float32, config["dt_config"], config["num_iters"],args.sim_id,db_addresses=args.db_addresses,ppn=config["ppn"])
