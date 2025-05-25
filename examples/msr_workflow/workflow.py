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

def main(ai_config_fname:str,sim_config_fname:str):

  # Load AI configuration
  with open(ai_config_fname, 'r') as f:
    ai_config = json.load(f)

  # Load simulation configuration
  with open(sim_config_fname, 'r') as f:
    sim_config = json.load(f)

  with open(os.getenv("PBS_NODEFILE"),"r") as f:
    nodes = [l.split(".")[0] for l in f.readlines()]

  db_nodes = nodes[ai_config["nnodes"]:]
  ai_nodes = nodes[:ai_config["nnodes"]]
  sim_nodes = nodes[ai_config["nnodes"]:]

  # db_nodes = nodes
  # ai_nodes = nodes
  # sim_nodes = nodes

  np_sim = sim_config['ppn']*sim_config['nnodes']
  nsims = len(sim_nodes)//sim_config["nnodes"]
  sim_cmds = []  
  ###launch AI on the head node.
  ai_cmd = f"mpirun -n {ai_config['ppn']*ai_config['nnodes']} -ppn {ai_config['ppn']}"+\
           f" --hosts {','.join(sim_nodes)} {ai_config['other_mpi_opts']}"+\
           f" python ai_exec.py --config {ai_config_fname} --nsims {nsims}"
  if ai_config["dt_config"]["type"] == "redis":
      ai_cmd += f" --db_addresses {','.join([f'{node}:6375' for node in db_nodes])}"
  
  for i in range(nsims):
    ns = i*sim_config["nnodes"]
    ne = (i+1)*sim_config["nnodes"]
    ##launch sim on all other nodes
    sim_cmd = f"mpirun -n {np_sim} -ppn {sim_config['ppn']} -hosts {','.join(sim_nodes[ns:ne])}"+\
              f" {sim_config['other_mpi_opts']} python sim_exec.py"+\
              f" --config {sim_config_fname} --sim_id {i}"
    if sim_config["dt_config"]["type"] == "redis":
      sim_cmd += f" --db_addresses {','.join([f'{node}:6375' for node in db_nodes])}"
    sim_cmds.append(sim_cmd)

  env = os.environ.copy()
  env["ZE_FLAT_DEVICE_HIERARCHY"]="COMPOSITE"

  if ai_config["dt_config"]["type"] == "redis":
    ##db is colocated with the AI nodes
    db_procs = []
    ###cmd for redis colocated db
    for node in db_nodes:
        db_cmd = f"mpirun -n 1 -ppn 1 -hosts {node}"+\
              f" {ai_config['dt_config']['redis-server-exe']} --port 6375 --bind 0.0.0.0 --protected-mode no"
        if ai_config["dt_config"]["type"] == "clustered":
          db_cmd += f" --cluster-enabled yes --cluster-config-file {nodes}.conf"
        p = subprocess.Popen(db_cmd, cwd=os.path.dirname(__file__), shell=True, env=env)
        db_procs.append(p)

    ##wait to db to start
    time.sleep(10)
    if ai_config["dt_config"]["type"] == "clustered":
      start = time.time()
      ##connect the clusters
      create_cmd = f"/home/ht1410/redis/src/redis-cli --cluster create " \
             f"{' '.join([f'{node}:6375' for node in db_nodes])} --cluster-replicas 0"
      subprocess.run(create_cmd, shell=True, check=True)
      print(f"It took {time.time()-start}s to start the cluster!")
      time.sleep(10)

  # # Wait for cluster to initialize
  time.sleep(5)

  sim_procs = []
  for i in range(nsims):
    p = subprocess.Popen(sim_cmds[i], cwd=os.path.dirname(__file__), shell=True, env=env)
    sim_procs.append(p)

  time.sleep(1) ##wait until the sim starts

  ai_process = subprocess.Popen(ai_cmd, cwd=os.path.dirname(__file__), shell=True, env=env)

  while ai_process.poll() is None or any(p.poll() is None for p in sim_procs):
    time.sleep(1)

  # Check the return codes
  ai_returncode = ai_process.returncode
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
  
  ##teardown the db
  if ai_config["dt_config"]["type"] == "redis":
    for p in db_procs:
      p.kill()

  # Return overall success/failure
  if ai_returncode == 0 and all(code == 0 for code in sim_returncodes):
    return 0
  else:
    return 1


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run the simulation and AI processes")
  parser.add_argument("--sim_config", type=str, default="sim_config.json", help="Path to JSON configuration file")
  parser.add_argument("--ai_config", type=str, default="ai_config.json", help="Path to JSON configuration file")
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
  
  main(args.ai_config,args.sim_config)