
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
from mpi4py import MPI
from wfMiniAPI.training import AI
import time
import argparse
import logging as logging_
import json
import socket


def main(device:str,nsteps_train:int,update_frequency:int,run_time:float,config:dict,nsims:int,db_addresses:str=None,ppn:int=None):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ###NOTE: this not ddp
    if config["type"] == "redis":
        assert db_addresses is not None and ppn is not None
        config["server-address"] = db_addresses.split(",")[rank//ppn]
    train_ai = AI(f"AI", config=config, num_hidden_layers=1, num_epochs=1, device=device,logging=(rank==0),log_level=logging_.INFO)
    comm.Barrier()
    for i in range(nsteps_train):
        tic = time.time()
        train_ai.train(run_time=run_time)
        train_time = time.time() - tic
        ##comsume the simulation data
        ## wait for the data to be staged
        if i%update_frequency == 0:
            tstart = time.time()
            for sim_id in range(nsims):
                while not train_ai.poll_staged_data(f"sim_{sim_id}_{rank}_{i//update_frequency}"):
                    if train_ai.logger:
                        train_ai.logger.debug(f"Waiting for data sim_{sim_id}_{rank}_{i//update_frequency}")
                        if time.time() - tstart > 10:
                            train_ai.logger.debug(f"Waiting for data sim_{sim_id}_{rank}_{i//update_frequency} timed out")
                    time.sleep(0.05)
                data = train_ai.stage_read(f"sim_{sim_id}_{rank}_{i//update_frequency}")
                train_ai.clean_staged_data(f"sim_{sim_id}_{rank}_{i//update_frequency}")
            dt_time = time.time() - tstart
        else:
            dt_time = 0.0
        
        if train_ai.logger:
            train_ai.logger.info(f"train time: {train_time}, dt time: {dt_time}")
        comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--config", type=str, required=True, help="help")
    parser.add_argument("--nsims", type=int, required=True, help="device")
    parser.add_argument("--db_addresses", type=str, default=None, help="help")
    args = parser.parse_args()
    with open(args.config,"r") as f:
        config = json.load(f)
    args = parser.parse_args()
    main(config["device"],
        config["num_iters"], 
        config["read_freq"], 
        config["run_time"], 
        config["dt_config"], 
        args.nsims,
        args.db_addresses,
        config["ppn"])
