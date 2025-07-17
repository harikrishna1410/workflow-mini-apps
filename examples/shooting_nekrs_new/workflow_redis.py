import subprocess
import os
import time
import argparse
import json
from utils import set_up_servers, split_nodes, clean_up


#********************************************************************************************

def launch_exec(exec_fname:str, config_fname:dict, nodes:list, db_addresses:list,env_vars:dict={}):
    with open(os.path.join(os.path.dirname(__file__), config_fname), 'r') as f:
        exec_config = json.load(f)
    ##
    np_exec = exec_config['ppn']*exec_config['nnodes']

    env = os.environ.copy()
    env.update(env_vars)

    env["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
    ##launch exec on all other nodes
    if os.getenv("PROFILE_WORKFLOW", None) is not None:
        ftof = {"sim_exec.py":"simulation.json","train_ai_exec.py":"training.json", "infer_ai_exec.py":"inference.json"}
        ftod = {"sim_exec.py":"simulation","train_ai_exec.py":"training", "infer_ai_exec.py":"inference"}
        out_dir = os.path.join(os.path.dirname(__file__), "profiles",f"{ftod[exec_fname]}")
        out_file = os.path.join(out_dir, f"{ftof[exec_fname]}")
        exec_cmd = f"mpirun -n {np_exec} -ppn {exec_config['ppn']} -hosts {','.join(nodes)}"+\
               f" {exec_config['other_mpi_opts']}"+\
               f" unitrace --chrome-kernel-logging --chrome-itt-logging --chrome-no-thread-on-device --chrome-no-engine-on-device --output-dir-path {out_dir}"+\
               f" python {exec_fname}"+\
               f" --config {config_fname}"
    else:
        exec_cmd = f"mpirun -n {np_exec} -ppn {exec_config['ppn']} -hosts {','.join(nodes)}"+\
               f" {exec_config['other_mpi_opts']} python {exec_fname}"+\
               f" --config {config_fname}"
    exec_cmd +=  f" --db_addresses {','.join(db_addresses)}"

    print(f"Launching command: {exec_cmd}")
    
    p = subprocess.Popen(exec_cmd, cwd=os.path.dirname(__file__), shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return p

#********************************************************************************************
#********************************************************************************************
def main(train_ai_config_fname:str, infer_ai_config_fname:str, sim_config_fname:str, ndb_per_node:int=1, placement:int=0):
    with open(os.path.join(os.path.dirname(__file__), train_ai_config_fname), 'r') as f:
        ai_config = json.load(f)
    ##clean up stuff
    clean_up(ai_config,[])
    
    ##split nodes
    db_nodes,ai_nodes,sim_nodes = split_nodes(train_ai_config_fname,method=placement)
    ##make 4 dbs per node
    db_addresses = []
    for node in db_nodes:
        for i in range(ndb_per_node):
            db_addresses.append(f"{node}:{6380+i}")

    ###set up redis db
    servers = set_up_servers(ai_config["dt_config"],db_addresses,ndb_per_node)

    #launch sim
    sim_proc = launch_exec("sim_exec.py",sim_config_fname,sim_nodes,db_addresses)
    #launch ai
    env_vars = {}
    env_vars["START_GPU"]="3"
    ai_proc = launch_exec("train_ai_exec.py", train_ai_config_fname, ai_nodes, db_addresses, env_vars)
    while ai_proc.poll() is None:
        time.sleep(1)
    # Check the return codes
    ai_returncode = ai_proc.returncode
    sim_returncode = sim_proc.returncode

    # Print status based on return codes
    if ai_returncode == 0:
        print("AI process completed successfully")
    else:
        clean_up(ai_config, servers)
        print(f"AI process failed with return code {ai_returncode}. stderr:{ai_proc.communicate()[-1]}")
        return 1

    if sim_returncode == 0:
        print(f"Simulation completed successfully")
    else:
        if sim_returncode is None:
            print("forcing simulation to stop")
            sim_proc.kill()
        else:
            clean_up(ai_config, servers)
            print(f"Simulation failed with return code {sim_returncode}. stderr:{sim_proc.communicate()[-1]}")
            return 1
        
    print("launching inference")
    infer_ai_proc = launch_exec("infer_ai_exec.py", infer_ai_config_fname, ai_nodes, db_addresses, env_vars)

    while infer_ai_proc.poll() is None:
        time.sleep(1)
    # Check the return codes
    infer_ai_returncode = infer_ai_proc.returncode
    # Return overall success/failure
    if infer_ai_returncode == 0:
        clean_up(ai_config, servers)
        return 0
    else:
        clean_up(ai_config, servers)
        print(f"Inference AI process failed with return code {infer_ai_returncode}. stderr:{infer_ai_proc.communicate()[-1]}")
        return 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the simulation and AI processes")
    parser.add_argument("--sim_config", type=str, default="configs/redis/sim_config.json", help="Path to JSON configuration file")
    parser.add_argument("--train_ai_config", type=str, default="configs/redis/train_ai_config.json", help="Path to JSON configuration file")
    parser.add_argument("--infer_ai_config", type=str, default="configs/redis/infer_ai_config.json", help="Path to JSON configuration file")
    parser.add_argument("--ndb_per_node",type=int,default=1)
    parser.add_argument("--placement",type=int,default=0)
    args = parser.parse_args()
    
    main(args.train_ai_config, args.infer_ai_config, args.sim_config, args.ndb_per_node, args.placement)