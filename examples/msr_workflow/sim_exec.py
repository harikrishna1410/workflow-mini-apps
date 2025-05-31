import mpi4py
mpi4py.rc.initialize=False
from mpi4py import MPI
from wfMiniAPI.simulation import Simulation as sim
import argparse
import numpy as np
import logging as logging_
import time
import json
import socket
import pickle

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

def main(write_freq:int,
         data_size:int,
         config:dict,
         niters:int,
         sim_id:int,
         init_MPI:bool=True,
         rank:int=0,
         size:int=1,
         ddict=None,
         db_addresses:str=None,
         ppn:int=None,
         dtype=np.float32):
    # Initialize MPI
    if init_MPI:
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    my_hostname = socket.gethostname()
    # Create a simulation object
    logging = rank==0
    if config["type"] == "redis":
        assert db_addresses is not None and ppn is not None
        config['role'] = "client"
        if config["db-type"] == "clustered":
            ##connect to all dbs on my node
            config["server-address"] = db_addresses
        else:
            local_dbs = [ad for ad in db_addresses.split(",") if my_hostname in ad]
            local_rank = rank%ppn
            ##assign in round robin fashion
            config["server-address"] = local_dbs[local_rank%len(local_dbs)]
            with open("sim_to_db.txt","a") as f:
                f.write(f"sim_{sim_id}_{rank} {local_dbs[local_rank%len(local_dbs)]}\n")
    elif config["type"] == "dragon":
        config["server-address"] = db_addresses
        config['role'] = "client"
        if ddict is not None:
            config["server-obj"] = ddict
        else:
            with open("server_obj.pickle", 'r') as f:
                config["server-obj"] = f.read()
    simulation = sim(name=f"sim_{sim_id}_{rank}", comm=(comm if init_MPI else None), config=config,logging=logging,log_level=logging_.DEBUG)

    # Initialize the simulation from a JSON file
    simulation.init_from_json("sim_telemetry.json")
    if init_MPI:
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
        i+=1
    simulation.stop_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True, help="Sim config")
    parser.add_argument("--sim_id", type=int, required=True, help="Sim index")
    parser.add_argument("--db_addresses", type=str, default=None)
    args = parser.parse_args()

    with open(args.config_name,"r") as f:
        config = json.load(f)    

    
    main(config["write_freq"], 
         config["data_size"], 
         config["dt_config"], 
         config["num_iters"],
         args.sim_id,
         db_addresses=args.db_addresses,
         ppn=config["ppn"])
