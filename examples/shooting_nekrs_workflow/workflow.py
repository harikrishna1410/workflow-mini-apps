from wfMiniAPI import Workflow, ServerManager
from sim_exec import main as sim_main
from train_ai_exec import main as train_ai_main
from infer_ai_exec import main as infer_ai_main
import os
import argparse
import json
import logging

def get_nodes():
    with open(os.getenv("PBS_NODEFILE", "/dev/null"), "r") as f:
        nodes = [line.split(".")[0] for line in f.readlines()]
    return nodes

def main():
    parser = argparse.ArgumentParser(description="Launch a workflow with a server component.")
    parser.add_argument("--server_config", type=str, required=True, help="Path to the server configuration file")
    parser.add_argument("--sim_config", type=str, default="configs/sim_config.json", help="Path to the simulation configuration file")
    parser.add_argument("--infer_ai_config", type=str, default="configs/infer_ai_config.json", help="Path to the inference AI configuration file")
    parser.add_argument("--train_ai_config", type=str, default="configs/train_ai_config.json", help="Path to the training AI configuration file")
    args = parser.parse_args()

    os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"
    my_workflow = Workflow(launcher={"mode":"mpi"}, sys_info={"name": "aurora", "ncores_per_node": 104, "ngpus_per_node": 12})

    ###get nodes
    nodes = get_nodes()
    if not nodes:
        raise RuntimeError("No nodes found in PBS_NODEFILE.")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    ##start server
    with open(os.path.join(root_dir, args.server_config), "r") as f:
        server_config = json.load(f)
    if server_config.get("type", "filesystem") == "redis" or server_config.get("type", "filesystem") == "dragon":
        server_config["server-address"] = ",".join([f"{n}:6875" for n in nodes])
    server = ServerManager("server", config=server_config, logging=True, log_level=logging.DEBUG)
    server.start_server()

    server_info = server.get_server_info()

    #start simulation component
    with open(os.path.join(root_dir, args.sim_config), "r") as f:
        sim_config = json.load(f)
    print(f"Simulation config: {sim_config}")
    my_workflow.register_component(
        name="sim",
        executable=sim_main,
        type="remote" if server_info["type"] != "dragon" else "dragon",
        args= {
            "sim_config": sim_config,
            "server_info": server_info,
        },
        nodes=nodes,
        ppn=6,  # 6 processes per node
        num_gpus_per_process=1,  # 1 GPU per process
        gpu_affinity=["0.0","0.1","1.0","1.1","2.0","2.1"]
    )

    #start training AI component
    with open(os.path.join(root_dir, args.train_ai_config), "r") as f:
        train_ai_config = json.load(f)

    my_workflow.register_component(
        name="train_ai",
        executable=train_ai_main,
        type="remote" if server_info["type"] != "dragon" else "dragon",
        args={
            "ai_config": train_ai_config,
            "server_info": server_info,
        },
        nodes=nodes,
        ppn=6,
        num_gpus_per_process=1,  # 1 GPU per process
        gpu_affinity=["3.0","3.1","4.0","4.1","5.0","5.1"]
    )

    #start inference AI component
    with open(os.path.join(root_dir, args.infer_ai_config), "r") as f:
        infer_ai_config = json.load(f)

    my_workflow.register_component(
        name="infer_ai",
        executable=infer_ai_main,
        type="remote" if server_info["type"] != "dragon" else "dragon",
        args={
            "ai_config": infer_ai_config,
            "server_info": server_info,
        },
        nodes=nodes,
        ppn=6,
        num_gpus_per_process=1,
        gpu_affinity=["3.0","3.1","4.0","4.1","5.0","5.1"],
        dependencies=["train_ai"]
    )

    # # # Run the workflow
    my_workflow.launch()

    ##stop the server
    server.stop_server()



if __name__ == "__main__":
    main()