import subprocess
import os,sys
import time
import argparse
import json
from utils import set_up_servers, split_nodes, clean_up
from train_ai_exec import main as train_ai_main
from infer_ai_exec import main as infer_ai_main
from sim_exec import main as sim_main
from wfMiniAPI import Component
try:
    import dragon
    from dragon.native.process_group import ProcessGroup
    from dragon.native.process import ProcessTemplate, Process, Popen
    from dragon.infrastructure.policy import Policy
    DRAGON_AVAILABLE = True
except Exception as e:
    raise ImportError("Dragon library is not available. Please install it to use this script.")


#********************************************************************************************
#********************************************************************************************
def launch_fn_dragon(fn,config_fname:dict,nodes:list,dragon_server:Component):
    with open(os.path.join(os.path.dirname(__file__), config_fname), 'r') as f:
        config = json.load(f)
    ###
    cpu_bind= [int(i) for i in config["cpu_bind"].split(",")]
    gpu_bind= [float(i) for i in config["gpu_bind"].split(",")]
    group_policy = Policy(distribution=Policy.Distribution.BLOCK)
    pg = ProcessGroup(pmi_enabled=False,policy=group_policy)
    for nid,node in enumerate(nodes):
        for local_rank in range(config["ppn"]):
            global_rank = nid * config["ppn"] + local_rank
            policy = Policy(placement=Policy.Placement.HOST_NAME,
                            host_name=node,
                            cpu_affinity=cpu_bind[local_rank:local_rank+1],
                            gpu_affinity=gpu_bind[local_rank:local_rank+1])
            env = os.environ.copy()
            env["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
            env["ZE_AFFINITY_MASK"]=gpu_bind[local_rank]
            fn_args = (
                config,
                False,  # init_MPI
                global_rank, # rank
                len(nodes) * config["ppn"], # size
                dragon_server.dragon_dict, #ddict
            )
            pg.add_process(
                    1,
                    template=ProcessTemplate(fn,args=fn_args,
                                             cwd=os.path.dirname(__file__),
                                             policy=policy,stdout=Popen.DEVNULL,env=env
                                            )
                    )
    pg.init()
    pg.start()
    return pg

#*****************************************************************************************

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
    sim_proc = launch_fn_dragon(sim_main,sim_config_fname,sim_nodes,servers[0])
    #launch ai
    ai_proc = launch_fn_dragon(train_ai_main, train_ai_config_fname, ai_nodes, servers[0])
    ai_proc.join()
    sim_proc.join()
    # Check the return codes
    ai_returncode = 1 if any(p[1]!=0 for p in ai_proc.inactive_puids) else 0
    sim_returncode = 1 if any(p[1]!=0 for p in sim_proc.inactive_puids) else 0
    ai_proc.stop()
    sim_proc.stop()
    # Print status based on return codes
    if ai_returncode == 0:
        print("AI process completed successfully")
    else:
        clean_up(ai_config, servers)
        print(f"AI process failed with return code {ai_returncode}.")
        return 1
    if sim_returncode == 0:
        print(f"Simulation completed successfully")
    else:
        clean_up(ai_config, servers)
        print(f"Simulation failed with return code {sim_returncode}.")
        return 1
    
    print("launching inference")
    infer_ai_proc = launch_fn_dragon(infer_ai_main, infer_ai_config_fname, ai_nodes, servers[0])
    infer_ai_proc.join()
    # Check the return codes
    infer_ai_returncode = 1 if any(p[1]!=0 for p in infer_ai_proc.inactive_puids) else 0
    infer_ai_proc.stop()
    print(f"Inference AI process completed with return code {infer_ai_returncode}")
    # Return overall success/failure
    if infer_ai_returncode == 0:
        clean_up(ai_config, servers)
        print("Cleaned up resources and exiting successfully")
        return 0
    else:
        clean_up(ai_config, servers)
        print(f"Inference AI process failed with return code {infer_ai_returncode}.")
        return 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the simulation and AI processes")
    parser.add_argument("--sim_config", type=str, default="configs/dragon/sim_config.json", help="Path to JSON configuration file")
    parser.add_argument("--train_ai_config", type=str, default="configs/dragon/train_ai_config.json", help="Path to JSON configuration file")
    parser.add_argument("--infer_ai_config", type=str, default="configs/dragon/infer_ai_config.json", help="Path to JSON configuration file")
    parser.add_argument("--ndb_per_node",type=int,default=1)
    parser.add_argument("--placement",type=int,default=0)
    args = parser.parse_args()
    
    return_code = main(args.train_ai_config, args.infer_ai_config, args.sim_config, args.ndb_per_node, args.placement)
    print("Workflow completed successfully")
    sys.exit(return_code)