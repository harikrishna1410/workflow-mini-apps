
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
from mpi4py import MPI
from wfMiniAPI.training import AI
import time
import argparse


def main(device:str,nsteps_train:int,update_frequency:int,config:dict):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ###NOTE: this not ddp
    train_ai = AI(f"train_AI_{rank}", config=config, neurons_per_layer=8, num_epochs=1, input_dim=8, output_dim=8, device=device,logging=(rank==0))
    ###set times
    train_ai.set_model_params_from_train_time(config["train_time"])
    for i in range(nsteps_train):
        tic = time.time()
        train_ai.train()
        train_time = time.time() - tic
        ##comsume the simulation data
        ## wait for the data to be staged
        if i%update_frequency == 0:
            tstart = time.time()
            while not train_ai.poll_staged_data(f"sim_data_{rank}_{i//update_frequency}"):
                if train_ai.logger:
                    train_ai.logger.debug(f"Waiting for data sim_data_{rank}_{i//update_frequency}")
                    if time.time() - tstart > 10:
                        train_ai.logger.debug(f"Waiting for data sim_data_{rank}_{i//update_frequency} timed out")
                time.sleep(0.05)
            data = train_ai.stage_read(f"sim_data_{rank}_{i//update_frequency}")
            train_ai.clean_staged_data(f"sim_data_{rank}_{i//update_frequency}")
            dt_time = time.time() - tstart
        else:
            dt_time = 0.0
        
        if train_ai.logger:
            train_ai.logger.info(f"train time: {train_time}, dt time: {dt_time}")
        comm.Barrier()
    
    data = train_ai.stage_write(f"ai_data_{rank}","kill_sim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--nsteps_train", type=int, required=True, help="Number of training steps")
    parser.add_argument("--location", type=str, required=True, help="Location for the data")
    parser.add_argument("--type", type=str, required=True, help="Transport type for the data")
    parser.add_argument("--train_time",type=float, required=True,help="Runtime for training")
    parser.add_argument("--update_frequency",type=int, required=True,help="Runtime for training")
    parser.add_argument("--device", type=str, required=True, help="device")
    args = parser.parse_args()
    config = {"type": args.type, 
              "location": args.location,  
              "train_time":args.train_time}
    main(args.device,args.nsteps_train, args.update_frequency, config)
