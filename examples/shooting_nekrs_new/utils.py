import subprocess
from wfMiniAPI.component import Component
import copy
import os
import time
import logging
import math
import pickle
import json
from glob import glob


def set_up_servers(config_in:dict,server_addresses:list,nservers_per_node:int):
    servers = []
    if config_in["type"] == "redis":
        dbtocpu = [1,8,16,24,32,40,53,60,68,76,84,92]
        env = os.environ.copy()
        for db_id,address in enumerate(server_addresses):
            config = copy.deepcopy(config_in)
            config["server-address"] = address
            config["role"] = "server"
            config["server-options"] = {}
            config["server-options"]["mpi-options"] = f"--cpu-bind list:{dbtocpu[db_id%nservers_per_node]}"
            ##this will automatically start the server
            servers.append(
                    Component(f"DB_{db_id}",config=config,logging=db_id==0,log_level=logging.DEBUG)
            )
        time.sleep(10)
        if config_in["db-type"] == "clustered":
            start = time.time()
            ##connect the clusters
            create_cmd = f"/home/ht1410/redis/src/redis-cli --cluster create " \
                        f"{' '.join(server_addresses)} --cluster-replicas 0 --cluster-yes"
            result = subprocess.run(create_cmd, shell=True, check=True, env=env, stdout=subprocess.DEVNULL)
            print(f"Cluster creation stderr: {result.stderr}")
            print(f"It took {time.time()-start}s to start the cluster!")
    elif config_in["type"] == "dragon":
        ##start the dragon dictionary
        config = copy.deepcopy(config_in)
        config["server-address"] = ",".join(server_addresses)
        config["role"] = "server"
        dragon_server = Component("db",config=config,logging=True)
        servers.append(dragon_server)
        time.sleep(10)
        # Pickle the db_obj for later use
        with open('server_obj.pickle', 'w') as pickle_file:
            pickle_file.write(dragon_server.dragon_dict.serialize())
    
    time.sleep(10)
    return servers


#method=0: db_nodes = sim_nodes = ai_nodes
##method=1: db_nodes = ai_nodes
##method=2: db_nodes = sim_nodes
##method=3: seperated db nodes
def split_nodes(ai_config_fname:str,method:int=2):
    with open(os.path.join(os.path.dirname(__file__), ai_config_fname), 'r') as f:
        ai_config = json.load(f)
    with open(os.getenv("PBS_NODEFILE"),"r") as f:
        nodes = [l.split(".")[0] for l in f.readlines()]

    if method == 0:
        ##all nodes are used for db, ai and sim
        return (nodes,nodes,nodes)
    ai_nodes = nodes[:ai_config["nnodes"]]
    sim_nodes = nodes[ai_config["nnodes"]:]
    db_nodes = []
    if ai_config["dt_config"]["type"] in ["redis","dragon"]:
        if method == 1:
            db_nodes = nodes[:ai_config["nnodes"]]  
        elif method == 2:
            db_nodes = nodes[ai_config["nnodes"]:]
        elif method == 3:
            ndb_nodes = int(2**math.ceil(math.log2(0.1*len(nodes))))
            ndb_nodes = max(1,ndb_nodes)
            db_nodes = nodes[:ndb_nodes]
            assert ndb_nodes + ai_config["nnodes"] <= len(nodes)
            ai_nodes = nodes[ndb_nodes : ndb_nodes + ai_config["nnodes"]]
            sim_nodes = nodes[ndb_nodes + ai_config["nnodes"]:]
        else:
            raise ValueError("Unknown splitting method")
    return (db_nodes,ai_nodes,sim_nodes)


#********************************************************************************************
#********************************************************************************************
def clean_up(ai_config:dict,servers:list):
    if ai_config["dt_config"]["type"] == "redis":
        fpath = "sim_to_db.txt"
        if os.path.exists(fpath):
            os.remove(fpath)
    elif ai_config["dt_config"]["type"] in ["filesystem","dragon"]:
        ##cleanup db's
        db_files = glob(os.path.join(ai_config["dt_config"].get("location","./.tmp"),"staging*"))
        for fpath in db_files:
            os.remove(fpath)
    for p in servers:
        p.stop_server()