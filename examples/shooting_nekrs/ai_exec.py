
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
from mpi4py import MPI
from wfMiniAPI.training import AI
import time
import argparse


def main(device:str,nsteps_train:int,nsteps_infer:int,update_frequency:int,config:dict):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ###NOTE: this not ddp
    infer_ai = AI(f"infer_AI_{rank}", config=config, neurons_per_layer=8,num_epochs=1, input_dim=8, output_dim=8, device=device,logging=(rank==0))
    train_ai = AI(f"train_AI_{rank}", config=config, neurons_per_layer=8, num_epochs=1, input_dim=8, output_dim=8, device=device,logging=(rank==0))
    if infer_ai.logger:
        infer_ai.logger.info(f"Set device to {device}")
    ###set times
    infer_ai.set_model_params_from_infer_time(config["infer_time"])
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
                time.sleep(1)
            data = train_ai.stage_read(f"sim_data_{rank}_{i//update_frequency}")
            train_ai.clean_staged_data(f"sim_data_{rank}_{i//update_frequency}")
            dt_time = time.time() - tstart
        else:
            dt_time = 0.0
        
        if train_ai.logger:
            train_ai.logger.info(f"train time: {train_time}, dt time: {dt_time}")
        comm.Barrier()
    
    data = train_ai.stage_write(f"ai_data_{rank}","kill_sim")
    ##infer
    for i in range(nsteps_infer):
        # Measure inference time
        tic_infer = time.time()
        infer_ai.infer()
        infer_time = time.time() - tic_infer
            
        # Measure write time
        tic_write = time.time()
        infer_ai.stage_write(f"ai_data_{rank}_{i}", data)
        write_time = time.time() - tic_write
            
        if infer_ai.logger:
            infer_ai.logger.info(f"infer time: {infer_time}, dt time: {write_time}")
                
        comm.Barrier()
    
    if rank%6 == 0:
        infer_ai.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--nsteps_train", type=int, required=True, help="Number of training steps")
    parser.add_argument("--nsteps_infer", type=int, required=True, help="Number of inference steps")
    parser.add_argument("--location", type=str, required=True, help="Location for the data")
    parser.add_argument("--type", type=str, required=True, help="Transport type for the data")
    parser.add_argument("--infer_time",type=float, required=True,help="Runtime for inference")
    parser.add_argument("--train_time",type=float, required=True,help="Runtime for training")
    parser.add_argument("--update_frequency",type=int, required=True,help="Runtime for training")
    parser.add_argument("--device", type=str, required=True, help="device")
    args = parser.parse_args()
    config = {"type": args.type, 
              "location": args.location, 
              "infer_time": args.infer_time, 
              "train_time":args.train_time}
    main(args.device,args.nsteps_train, args.nsteps_infer, args.update_frequency, config)
