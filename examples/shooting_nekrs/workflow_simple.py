import subprocess
import os
import sys
import time
import argparse
import shutil
import json

"""
This is example workflow from [1]
[1] https://github.com/argonne-lcf/nekRS-ML/tree/main/examples/shooting_workflow_smartredis
"""

def main(np:int,
                other_mpi_opts_sim:str,
                other_mpi_opts_ai:str,
                nsteps_train:int,
                nsteps_infer:int,
                model_update_freq:int,
                write_freq:int,
                dt_type:str,
                dt_location:str,
                infer_time:float,
                train_time:float,
                ai_device:str,
                staging_data_size:int):
        sim_cmd = f"mpirun -n {np} {other_mpi_opts_sim} python sim_exec.py"+\
                        f" --write_freq {write_freq}"+\
                        f" --location {dt_location} --type {dt_type} --data_size {staging_data_size}"
    
        train_ai_cmd = f"mpirun -n {np} {other_mpi_opts_ai}"+\
                        f" python train_ai_exec.py --nsteps_train {nsteps_train}"+\
                        f" --update_frequency {model_update_freq}"+\
                        f" --location {dt_location} --type {dt_type}"+\
                        f" --train_time {train_time} --device {ai_device}"

        env_sim = os.environ.copy()
        env_ai = os.environ.copy()
        env_ai["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
        env_sim["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
        env_ai["START_GPU"]="3"
        sim_process = subprocess.Popen(sim_cmd, cwd=os.path.dirname(__file__), shell=True, env=env_sim)
        while sim_process.poll() is None:
        time.sleep(1)

        # time.sleep(1) ##wait until the sim starts
        # train_ai_process = subprocess.Popen(train_ai_cmd, cwd=os.path.dirname(__file__), shell=True, env=env_ai)

        # while train_ai_process.poll() is None:
        #   if sim_process.poll() is not None:
        #     print("Error: simulation failed!!")
        #     train_ai_process.kill()
        #     break
        #   time.sleep(1)

        # returncode = train_ai_process.poll()
        # if sim_process.poll() is None:
        #   sim_process.kill()
        #   if returncode == 0:
        #     print("Training successful")
        #   else:
        #     print("Training failed")

        # infer_ai_cmd = f"mpirun -n {np} {other_mpi_opts_ai}"+\
        #           f" python infer_ai_exec.py "+\
        #           f" --nsteps_infer {nsteps_infer}"+\
        #           f" --location {dt_location} --type {dt_type}"+\
        #           f" --infer_time {infer_time} --device {ai_device} --data_size {staging_data_size}"
        # infer_ai_process = subprocess.Popen(infer_ai_cmd, cwd=os.path.dirname(__file__), shell=True, env=env_ai)
        # while infer_ai_process.poll() is None:
        #   time.sleep(1)
    
        # returncode = infer_ai_process.poll()
        # if returncode != 0:
        #   print("Inference failed")
        # else:
        #   print("Inference successful")
        # return


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run the simulation and AI processes")
        parser.add_argument("--config", type=str, default="config.json", help="Path to JSON configuration file")
        args = parser.parse_args()
    
        # Default configuration
        config = {
        "np": 4,
        "other_mpi_opts_sim": "",
        "other_mpi_opts_ai": "",
        "nsteps_train": 100,
        "nsteps_infer": 10,
        "model_update_freq": 10,
        "sim_write_freq": 100,
        "dt_type": "filesystem",
        "dt_location": os.path.join(os.getcwd(), ".tmp"),
        "infer_time": 0.02,
        "train_time": 0.07,
        "ai_device":"cpu",
        "staging_data_size":32*32*32
        }
    
        # Load configuration from JSON file
        try:
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
        except FileNotFoundError:
        print(f"Config file {args.config} not found. Using default configuration.")
        except json.JSONDecodeError:
        print(f"Error parsing config file {args.config}. Using default configuration.")
    
        if config["dt_type"] == "filesystem":
        if os.path.exists(config["dt_location"]):
            shutil.rmtree(config["dt_location"])
        os.makedirs(config["dt_location"], exist_ok=True)
    
        logs = os.path.join(os.path.dirname(__file__),"logs")
        if os.path.exists(logs):
        shutil.rmtree(logs)
    
    
        main(np=config["np"], 
            other_mpi_opts_sim=config["other_mpi_opts_sim"],
            other_mpi_opts_ai=config["other_mpi_opts_ai"], 
            nsteps_train=config["nsteps_train"], 
            nsteps_infer=config["nsteps_infer"],
            model_update_freq=config["model_update_freq"],
            write_freq=config["sim_write_freq"], 
            dt_type=config["dt_type"], 
            dt_location=config["dt_location"],
            infer_time=config["infer_time"],
            train_time=config["train_time"],
            ai_device=config["ai_device"],
            staging_data_size=config["staging_data_size"])
