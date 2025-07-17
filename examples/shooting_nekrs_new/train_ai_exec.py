
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from wfMiniAPI.training import AI
import time
import argparse
import logging as logging_
import json
import socket
import os
import pyitt


def main(ai_config:str,
         init_MPI=True,
         rank:int=0,
         size:int=1,
         ddict=None,
         db_addresses:str=None,
         init_time:float=12.3):
    device = ai_config["device"]
    nsteps = ai_config["num_iters"]
    update_frequency = ai_config["read_freq"]
    run_time = ai_config["run_time"]
    config = ai_config["dt_config"]
    config["role"] = "client"
    config["server-address"] = db_addresses
    config["server-obj"] = ddict
    nrequests = ai_config.get("nrequests", 1)
    # Initialize MPI
    if init_MPI:
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    train_ai = AI(f"train_ai_{rank}", config=config, num_hidden_layers=1, num_epochs=1, 
                    device=device,logging=rank==0,log_level=logging_.DEBUG,data_size=2)
    if init_MPI:
        comm.Barrier()
    last_update = 0
        
    # Wait for the first file to be available before starting the loop
    if train_ai.logger:
        train_ai.logger.info(f"Waiting for first simulation data file: sim_output_{rank}_0")
    while not train_ai.poll_staged_data(f"sim_output_{rank}_0"):
        time.sleep(0.1)  # Small delay to avoid busy waiting
    last_update += 1
    if train_ai.logger:
        train_ai.logger.info(f"First simulation data file detected, starting training loop")
    
    warmup_time = time.time()
    train_ai.train(run_time=run_time)
    warmup_time = time.time() - warmup_time
    if warmup_time < init_time:
        time.sleep(init_time-warmup_time)

    for i in range(nsteps):
        tic = time.time()
        if os.getenv("PROFILE_WORKFLOW",None) is not None:
            with pyitt.task(f"inference_step_{i}",domain="training"):
                elap_time, rc = train_ai.train(run_time=run_time)
        else:
            elap_time, rc = train_ai.train(run_time=run_time)
        train_time = time.time() - tic
        ##comsume the simulation data
        ## wait for the data to be staged
        read_data = False
        if i%update_frequency == 0 and i > 0:
            nread = 0
            tstart = time.time()
            dt_time = 0.0
            ###read until all staged data is read
            while train_ai.poll_staged_data(f"sim_output_{rank}_{last_update}"):
                tic = time.time()
                data = train_ai.stage_read(f"sim_input_{rank}_{last_update}")
                data = train_ai.stage_read(f"sim_output_{rank}_{last_update}")
                if train_ai.logger:
                    train_ai.logger.info(f"Read data: sim_output_{rank}_{last_update}")
                dt_time += time.time() - tic
                last_update += 1
                read_data = True
                # train_ai.cleanup_staged_data(f"sim_input_{rank}_{last_update}")
                # train_ai.cleanup_staged_data(f"sim_output_{rank}_{last_update}")
            dt_time_total = time.time() - tstart
        else:
            dt_time_total = 0.0
            dt_time = 0.0
        if not read_data:
            dt_time_total = 0.0
            dt_time = 0.0
        if train_ai.logger:
            train_ai.logger.info(f"tstep time: {train_time}, dt time: {dt_time}")
        if init_MPI:
            comm.Barrier()
    train_ai.stage_write(f"ai_data_{rank}", "kill_sim")
    train_ai.stop_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--config", type=str, required=True, help="help")
    parser.add_argument("--db_addresses", type=str, default=None, help="help")
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(__file__), args.config), "r") as f:
        config = json.load(f)
    args = parser.parse_args()
    main(config, 
        db_addresses=args.db_addresses)
