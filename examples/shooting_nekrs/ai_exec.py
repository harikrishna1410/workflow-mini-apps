
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
    infer_ai = AI(f"infer_AI_{rank}", config=config, neurons_per_layer=32,num_epochs=1, input_dim=32, output_dim=32, device=device)
    train_ai = AI(f"train_AI_{rank}", config=config, neurons_per_layer=32, num_epochs=1, input_dim=32, output_dim=32, device=device)
    ###set times
    infer_ai.set_model_params_from_infer_time(config["infer_time"])
    train_ai.set_model_params_from_train_time(config["train_time"])
    for i in range(nsteps_train):
        train_ai.logger.info(f"Timestep {i}")
        train_ai.train()
        ##comsume the simulation data
        ## wait for the data to be staged
        if i%update_frequency == 0:
            tstart = time.time()
            while not train_ai.poll_staged_data(f"sim_data_{rank}_{i//update_frequency}"):
                train_ai.logger.info(f"Waiting for data sim_data_{rank}_{i//update_frequency}")
                if time.time() - tstart > 10:
                    train_ai.logger.info(f"Waiting for data sim_data_{rank}_{i//update_frequency} timed out")
                time.sleep(1)
            data = train_ai.stage_read(f"sim_data_{rank}_{i//update_frequency}")
            train_ai.clean_staged_data(f"sim_data_{rank}_{i//update_frequency}")
        comm.Barrier()
    
    data = train_ai.stage_write(f"ai_data_{rank}","kill_sim")
    ##infer
    for i in range(nsteps_infer):
        infer_ai.logger.info(f"Timestep {i}")
        infer_ai.infer()
        infer_ai.stage_write(f"ai_data_{rank}_{i}", data)
        comm.Barrier()


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
