import subprocess
import os
import sys
import time
import argparse
import shutil
import json
from wfMiniAPI.component import Component
from sim_exec import main as sim_main
from ai_exec import main as ai_main
import copy
import math
import logging
from glob import glob

try:
  import dragon
  from dragon.native.process_group import ProcessGroup
  from dragon.native.process import ProcessTemplate, Process, Popen
  from dragon.infrastructure.policy import Policy
  DRAGON_AVAILABLE = True
except:
  DRAGON_AVAILABLE = False

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
            Component(f"DB_{db_id}",config=config,logging=True,log_level=logging.DEBUG)
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
    else:
      raise ValueError("Unknown type")
    time.sleep(10)
    return servers
#********************************************************************************************
#********************************************************************************************
def launch_ai_mpi(ai_config_fname:str,sim_config_fname:str,ai_nodes:list,sim_nodes:list,server_addresses:list):
  with open(ai_config_fname, 'r') as f:
    ai_config = json.load(f)

  with open(sim_config_fname, 'r') as f:
    sim_config = json.load(f)
  ###
  nsims = len(sim_nodes)//sim_config["nnodes"]
  ai_cmd = f"mpirun -n {ai_config['ppn']*ai_config['nnodes']} -ppn {ai_config['ppn']}"+\
           f" --hosts {','.join(ai_nodes)} {ai_config['other_mpi_opts']}"+\
           f" python ai_exec.py --config {ai_config_fname} --nsims {nsims}"
  if ai_config["dt_config"]["type"] == "redis":
      ai_cmd += f" --db_addresses {','.join(server_addresses)}"
  
  env = os.environ.copy()
  env["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
  ai_process = subprocess.Popen(ai_cmd, cwd=os.path.dirname(__file__), shell=True, env=env)
  return [ai_process]
#********************************************************************************************
#********************************************************************************************
def launch_ai_dragon(ai_config_fname:str,sim_config_fname:str,ai_nodes:list,sim_nodes:list,dragon_server:Component):
  with open(ai_config_fname, 'r') as f:
    ai_config = json.load(f)

  with open(sim_config_fname, 'r') as f:
    sim_config = json.load(f)
  ###
  cpu_bind=[1,8,16,24,32,40,53,60,68,76,84,92]
  gpu_bind=["0.0","0.1","1.0","1.1","2.0","2.1","3.0","3.1","4.0","4.1","5.0","5.1"]
  nsims = len(sim_nodes)//sim_config["nnodes"]
  ai_group_policy = Policy(distribution=Policy.Distribution.BLOCK)
  ai_pg = ProcessGroup(pmi_enabled=False,policy=ai_group_policy)
  for nid,node in enumerate(ai_nodes):
    for local_rank in range(ai_config["ppn"]):
      policy = Policy(
                        placement=Policy.Placement.HOST_NAME,
                        host_name=node,
                        cpu_affinity=cpu_bind[local_rank:local_rank+1],
                        gpu_affinity=gpu_bind[local_rank:local_rank+1]
                    )
      env = os.environ.copy()
      env["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
      env["ZE_AFFINITY_MASK"]=gpu_bind[local_rank]
      ai_pg.add_process(
        1,
        template=ProcessTemplate(ai_main,args=(
                                                ai_config["device"],
                                                ai_config["num_iters"],
                                                ai_config["read_freq"],
                                                ai_config["run_time"],
                                                ai_config["dt_config"],
                                                nsims,
                                                "",##db addresses
                                                ai_config["ppn"],
                                                False, ##init_MPI
                                                nid*ai_config["ppn"]+local_rank, ##rank
                                                ai_config["nnodes"]*ai_config["ppn"], ##size
                                                dragon_server.dragon_dict
                                              ),
                                    cwd=os.path.dirname(__file__),
                                    policy=policy,stdout=Popen.DEVNULL,env=env
                                )
        )
  ai_pg.init()
  ai_pg.start()
  return [ai_pg]
#********************************************************************************************
#********************************************************************************************
def launch_simulations_mpi(sim_config_fname:str,sim_nodes:list,server_addresses:list):
  # Load simulation configuration
  with open(sim_config_fname, 'r') as f:
    sim_config = json.load(f)
  ##
  np_sim = sim_config['ppn']*sim_config['nnodes']
  nsims = len(sim_nodes)//sim_config["nnodes"]
  assert nsims > 0
  sim_procs = []  
  env = os.environ.copy()
  env["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
  for sim_id in range(nsims):
    ns = sim_id*sim_config["nnodes"]
    ne = (sim_id+1)*sim_config["nnodes"]
    ##launch sim on all other nodes
    sim_cmd = f"mpirun -n {np_sim} -ppn {sim_config['ppn']} -hosts {','.join(sim_nodes[ns:ne])}"+\
              f" {sim_config['other_mpi_opts']} python sim_exec.py"+\
              f" --config {sim_config_fname} --sim_id {sim_id}"
    if sim_config["dt_config"]["type"] == "redis":
      sim_cmd += f" --db_addresses {','.join(server_addresses)}"
    
    p = subprocess.Popen(sim_cmd, cwd=os.path.dirname(__file__), shell=True, env=env)
    sim_procs.append(p)
  return sim_procs

#********************************************************************************************
#********************************************************************************************
def launch_simulations_dragon(sim_config_fname:str,sim_nodes:list,dragon_server):
  # Load simulation configuration
  with open(sim_config_fname, 'r') as f:
    sim_config = json.load(f)
  ##
  np_sim = sim_config['ppn']*sim_config['nnodes']
  nsims = len(sim_nodes)//sim_config["nnodes"]
  assert nsims > 0
  sim_pgs = []
  cpu_bind=[1,8,16,24,32,40,53,60,68,76,84,92]
  gpu_bind=["0.0","0.1","1.0","1.1","2.0","2.1","3.0","3.1","4.0","4.1","5.0","5.1"]
  for sim_id in range(nsims):
    sim_pg = ProcessGroup(pmi_enabled=False,policy=Policy(distribution=Policy.Distribution.BLOCK))
    ns = sim_id*sim_config["nnodes"]
    ne = (sim_id+1)*sim_config["nnodes"]
    for nid,node in enumerate(sim_nodes[ns:ne]):
      for local_rank in range(sim_config["ppn"]):
        policy = Policy(
                        placement=Policy.Placement.HOST_NAME,
                        host_name=node,
                        cpu_affinity=cpu_bind[local_rank:local_rank+1],
                        gpu_affinity=gpu_bind[local_rank:local_rank+1]
                    )
        env = os.environ.copy()
        env["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"
        env["ZE_AFFINITY_MASK"]=gpu_bind[local_rank]
        sim_pg.add_process(
                            nproc=1,
                            template=ProcessTemplate(target=sim_main, args=(
                                                          sim_config["write_freq"],
                                                          sim_config["data_size"],
                                                          sim_config["dt_config"],
                                                          sim_config["num_iters"],
                                                          sim_id,
                                                          False,##init_MPI
                                                          nid*sim_config["ppn"]+local_rank,##rank
                                                          sim_config["ppn"]*sim_config["nnodes"],
                                                          dragon_server.dragon_dict,
                                                         ), 
                                                    cwd=os.path.dirname(__file__), 
                                                    stdout=Popen.DEVNULL,
                                                    policy=policy,
                                                    env=env)
                            )
      
    sim_pgs.append(sim_pg)
    sim_pg.init()
    sim_pg.start()
  return sim_pgs

##method=1: db_nodes = ai_nodes
##method=2: db_nodes = sim_nodes
##method=3: seperated db nodes
def split_nodes(ai_config_fname:str,method:int=2):
    with open(ai_config_fname, 'r') as f:
      ai_config = json.load(f)

    with open(os.getenv("PBS_NODEFILE"),"r") as f:
      nodes = [l.split(".")[0] for l in f.readlines()]
    
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
def clean_up(ai_config:dict):

  if ai_config["dt_config"]["type"] == "redis":
    fpath = os.path.join(os.path.dirname(__file__),"sim_to_db.txt")
    if os.path.exists(fpath):
      os.remove(fpath)
  elif ai_config["dt_config"]["type"] in ["filesystem","dragon"]:
    ##cleanup db's
    db_files = glob(os.path.join(ai_config["dt_config"].get("location","./.tmp"),"staging*"))
    for fpath in db_files:
      os.remove(fpath)
  logfiles = glob(os.path.join("logs","*.log"))
  for fpath in logfiles:
      os.remove(fpath)
  

#********************************************************************************************
#********************************************************************************************
def main(ai_config_fname:str,sim_config_fname:str,ndb_per_node:int=1,placement:int=2):
  with open(ai_config_fname, 'r') as f:
    ai_config = json.load(f)

  ##clean up stuff
  clean_up(ai_config)
  # Load AI configuration
  db_nodes,ai_nodes,sim_nodes = split_nodes(ai_config_fname,method=placement)
  ##make 4 dbs per node
  db_addresses = []
  for node in db_nodes:
    for i in range(ndb_per_node):
      db_addresses.append(f"{node}:{6375+i}")

  if ai_config["dt_config"]["type"] in ["redis","dragon"]:
    servers = set_up_servers(ai_config["dt_config"],db_addresses,ndb_per_node)

  if ai_config["dt_config"]["type"] == "dragon":
    sim_pgs = launch_simulations_dragon(sim_config_fname,sim_nodes,servers[0])
    ai_pg = launch_ai_dragon(ai_config_fname,sim_config_fname,ai_nodes,sim_nodes,servers[0])[0]
    ai_pg.join()
    ai_pg.close()
    ##wait
    for pg in sim_pgs:
      try:
        pg.join(30)
      except Exception as e:
        pg.stop()
      pg.close()
  else:
    sim_procs = launch_simulations_mpi(sim_config_fname,sim_nodes,db_addresses)
    time.sleep(30)
    ai_procs = launch_ai_mpi(ai_config_fname,sim_config_fname,ai_nodes,sim_nodes,db_addresses)

    while ai_procs[0].poll() is None or any(p.poll() is None for p in sim_procs):
      time.sleep(1)

    # Check the return codes
    ai_returncode = ai_procs[0].returncode
    sim_returncodes = [p.returncode for p in sim_procs]
  
    # Print status based on return codes
    if ai_returncode == 0:
      print("AI process completed successfully")
    else:
      print(f"AI process failed with return code {ai_returncode}")
  
    for i, returncode in enumerate(sim_returncodes):
      if returncode == 0:
        print(f"Simulation {i+1} completed successfully")
      else:
        print(f"Simulation {i+1} failed with return code {returncode}")
  
    if ai_config["dt_config"]["type"] in ["redis","dragon"]:
      ##teardown the db
      for p in servers:
        p.stop_server()

    # Return overall success/failure
    if ai_returncode == 0 and all(code == 0 for code in sim_returncodes):
      return 0
    else:
      return 1


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run the simulation and AI processes")
  parser.add_argument("--sim_config", type=str, default="sim_config.json", help="Path to JSON configuration file")
  parser.add_argument("--ai_config", type=str, default="ai_config.json", help="Path to JSON configuration file")
  parser.add_argument("--ndb_per_node",type=int,default=1)
  parser.add_argument("--placement",type=int,default=2)
  args = parser.parse_args()
  
  
  # Load AI and simulation configurations and check for missing keys
  try:
    with open(args.ai_config, 'r') as f:
      ai_config = json.load(f)
      missing_keys = [key for key in ["ppn", "nnodes", "other_mpi_opts", "run_time", "num_iters", "dt_config", "read_freq", "data_size", "device"] if key not in ai_config]
      if missing_keys:
        print(f"Error: Missing AI config options: {missing_keys}")
        sys.exit(1)
  except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading AI config file: {e}")
    sys.exit(1)
    
  try:
    with open(args.sim_config, 'r') as f:
      sim_config = json.load(f)
      missing_keys = [key for key in ["ppn", "nnodes", "other_mpi_opts", "run_time", "num_iters", "dt_config", "write_freq", "data_size"] if key not in sim_config]
      if missing_keys:
        print(f"Warning: Missing keys in simulation config: {missing_keys}")
  except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading simulation config file: {e}")
    sys.exit(1)
  
  assert sim_config["ppn"] == ai_config["ppn"] and sim_config["nnodes"] == ai_config["nnodes"]
  assert ai_config["dt_config"]["type"] == sim_config["dt_config"]["type"]
  
  main(args.ai_config,args.sim_config,args.ndb_per_node,args.placement)