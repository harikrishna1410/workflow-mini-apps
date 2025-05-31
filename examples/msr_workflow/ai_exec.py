
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


def main(device:str,
         nsteps_train:int,
         update_frequency:int,
         run_time:float,
         config:dict,
         nsims:int,
         db_addresses:str=None,
         ppn:int=None,
         init_MPI=True,
         rank:int=0,
         size:int=1,
         ddict=None):
    # Initialize MPI
    if init_MPI:
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    ###NOTE: this not ddp
    if config["type"] == "redis":
        assert db_addresses is not None and ppn is not None
        config["server-address"] = db_addresses
        config['role'] = "client"
        if config['db-type'] == "colocated":
            fpath = os.path.join(os.path.dirname(__file__),"sim_to_db.txt")
            ##get a sim name to client index dict
            with open(fpath,"r") as f:
                lines = f.readlines()
            dbs = db_addresses.split(",")
            sim_to_client = {}
            for line in lines:
                sim_to_client[line.split(" ")[0]] = dbs.index((line.split(" ")[1]).strip())
    elif config["type"] == "dragon":
        config['role'] = "client"
        ##thid doesn't matter for dragon
        config["server-address"] = db_addresses
        if ddict is not None:
            config["server-obj"] = ddict
        else:
            with open("server_obj.pickle", 'r') as f:
                config["server-obj"] = f.read()
    train_ai = AI(f"AI_{rank}", config=config, num_hidden_layers=1, num_epochs=1, device=device,logging=rank==0,log_level=logging_.INFO)
    if init_MPI:
        comm.Barrier()
    for i in range(nsteps_train):
        tic = time.time()
        elap_time = train_ai.train(run_time=run_time)
        train_time = time.time() - tic
        ##comsume the simulation data
        ## wait for the data to be staged
        if i%update_frequency == 0:
            nread = 0
            tstart = time.time()
            dt_time = 0.0
            while nread != nsims and  time.time() - tstart < 10:
                for sim_id in range(nsims):
                    if train_ai.poll_staged_data(f"sim_{sim_id}_{rank}_{i//update_frequency}",client_id=sim_to_client[f"sim_{sim_id}_{rank}"] if config["type"] == "redis" and config["db-type"] == "colocated" else 0):
                        tic = time.time()
                        data = train_ai.stage_read(f"sim_{sim_id}_{rank}_{i//update_frequency}",client_id=sim_to_client[f"sim_{sim_id}_{rank}"] if config["type"] == "redis" and config["db-type"] == "colocated" else 0)
                        train_ai.clean_staged_data(f"sim_{sim_id}_{rank}_{i//update_frequency}",client_id=sim_to_client[f"sim_{sim_id}_{rank}"] if config["type"] == "redis" and config["db-type"] == "colocated" else 0)
                        dt_time += time.time() - tic
                        nread+=1
                time.sleep(1.0)
            dt_time_total = time.time() - tstart
            if nread != nsims:
                if train_ai.logger:
                    train_ai.logger.critical(f"Only read {nread}/{nsims} simulation data.")
        else:
            dt_time_total = 0.0
            dt_time = 0.0
        
        if train_ai.logger:
            train_ai.logger.info(f"train time: {train_time}, returned time:{elap_time}, total dt time: {dt_time_total}, actual dt time: {dt_time}, nsims read: {nread}")
        if init_MPI:
            comm.Barrier()
    train_ai.stop_client()


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
