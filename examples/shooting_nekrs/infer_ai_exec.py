
"""
This is the AI model of the whole simulations. However, to support weak scaling, each MPI process
creates its own AI model. It is assumed that the training is model parallel.
"""
from mpi4py import MPI
from wfMiniAPI.training import AI
import time
import argparse
import numpy as np


def main(device:str,nsteps_infer:int,config:dict,data_size:int):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ###NOTE: this not ddp
    infer_ai = AI(f"infer_AI_{rank}", config=config, num_layers=32, device=device,logging=(rank==0))
    ##infer
    data = np.empty(data_size)
    for i in range(nsteps_infer):
        # Measure inference time
        tic_infer = time.time()
        infer_ai.infer(run_time=config["infer_time"])
        infer_time = time.time() - tic_infer
            
        # Measure write time
        tic_write = time.time()
        infer_ai.stage_write(f"ai_data_{rank}_{i}", data)
        write_time = time.time() - tic_write
            
        if infer_ai.logger:
            infer_ai.logger.info(f"infer time: {infer_time}, dt time: {write_time}")
                
        comm.Barrier()
    
    if rank == 0:
        infer_ai.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI training")
    parser.add_argument("--nsteps_infer", type=int, required=True, help="Number of inference steps")
    parser.add_argument("--location", type=str, required=True, help="Location for the data")
    parser.add_argument("--type", type=str, required=True, help="Transport type for the data")
    parser.add_argument("--infer_time",type=float, required=True,help="Runtime for inference")
    parser.add_argument("--device", type=str, required=True, help="device")
    parser.add_argument("--data_size",type=int, required=True,help="Data size to stage")
    args = parser.parse_args()
    config = {"type": args.type, 
              "location": args.location, 
              "infer_time": args.infer_time, }
    main(args.device, args.nsteps_infer, config, args.data_size)
