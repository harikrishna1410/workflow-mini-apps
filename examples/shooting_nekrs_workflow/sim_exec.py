import mpi4py
mpi4py.rc.initialize=False
from mpi4py import MPI
from wfMiniAPI.simulation import Simulation as sim
import argparse
import numpy as np
import logging as logging_
import time
import json
import os
import pyitt
from typing import Union


def main(sim_config:dict,
         server_info:Union[str, dict]=None,
         init_MPI:bool=True,
         rank:int=0,
         size:int=1,
         dtype=np.float32,
         init_time:float=31.0309):
    # Initialize MPI
    if init_MPI:
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    ##
    data_size = sim_config["data_size"]
    write_freq = sim_config["write_freq"]
    sim_telemetry = sim_config["sim_telemetry"]
    nrequests = sim_config.get("nrequests", 1)

    # Create a simulation object
    simulation = sim(name=f"sim_{rank}", 
                     comm=(comm if init_MPI else None), 
                     server_info=server_info,
                     logging=rank==0,
                     log_level=logging_.INFO,
                     size=size, 
                     rank=rank)
    # Initialize the simulation from a JSON file
    simulation.init_from_json(os.path.join(os.path.dirname(__file__), sim_telemetry))
    if simulation.logger:
        simulation.logger.info(f"Simulation initialized with name {simulation.name}, rank {rank}, size {size}")
    if init_MPI:
        comm.Barrier()
    i=0
    warmup_time = time.time()
    simulation.run()
    warmup_time = time.time() - warmup_time
    if warmup_time < init_time:
        time.sleep(init_time-warmup_time) 
    while True:
        tic = time.time()
        # Run the simulation step
        if os.getenv("PROFILE_WORKFLOW", None) is not None:
            with pyitt.task(f"simulation_step_{i}",domain="simulation"):
                iter_dt_out = simulation.run(nsteps=1)
        else:
            iter_dt_out = simulation.run(nsteps=1)
        iter_time = time.time() - tic
        # Stage data for the AI to read
        if i % write_freq == 0 or i % write_freq == 10:
            if i % write_freq == 0:
                fname_ext = "input"
            else:
                fname_ext = "output"
            if simulation.logger:
                simulation.logger.info(f"Write the data: sim_{fname_ext}_{rank}_{i//write_freq}")
            tic = time.time()
            simulation.stage_write(f"sim_{fname_ext}_{rank}_{i//write_freq}", np.empty(data_size, dtype=dtype))
            toc = time.time()
            data_write_time = toc - tic
        else:
            data_write_time = 0.0
        
        if simulation.logger:
            simulation.logger.info(f"tstep time: {iter_time}, dt time: {data_write_time}")

        if simulation.poll_staged_data(f"ai_data_{rank}"):
            data = simulation.stage_read(f"ai_data_{rank}")
            if data=="kill_sim":
                if simulation.logger:
                    simulation.logger.info("Received kill message!")
                break
        i+=1

    simulation.flush_logger()
    simulation.clean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Sim config")
    parser.add_argument("--server_info", type=str, default=None, help="Server info for the simulation")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.config), "r") as f:
        config = json.load(f)

    main(config, server_info=args.server_info)
