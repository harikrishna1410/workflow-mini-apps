"""
Component module for workflow mini-apps providing data storage and server management.

Complete Workflow Examples:

1. Filesystem-based Workflow:
```python
# Server side
server_config = {"type": "filesystem", "server-address": "./data", "nshards": 32}
server = ServerManager("fs_server", server_config)
server.start_server()

# Client side - use server info directly
server_info = server.get_server_info()
datastore = DataStore("worker1", server_info)
# OR use serialized string
serialized = server.serialize()
datastore = DataStore("worker1", serialized)

datastore.stage_write("results", {"accuracy": 0.95})
data = datastore.stage_read("results")
```

2. Redis Server + Client Workflow (Colocated):
```python
# Server side
server_config = {
    "type": "redis",
    "db-type": "colocated",
    "server-address": "localhost:6379",
    "redis-server-exe": "/usr/bin/redis-server"
}
server = ServerManager("redis_server", server_config)
server.start_server()

# Client side - use server info directly
server_info = server.get_server_info()
datastore = DataStore("worker1", server_info)
# OR use serialized string for remote clients
serialized = server.serialize()
datastore = DataStore("worker1", serialized)

datastore.stage_write("model_weights", weights_data)
```

3. Redis Cluster Workflow:
```python
# Start multiple Redis servers first, then create cluster
server_addresses = ["node1:6379", "node2:6379", "node3:6379"]
ServerManager.create_redis_cluster(server_addresses)

# Server side
server_config = {
    "type": "redis",
    "db-type": "clustered",
    "server-address": "node1:6379,node2:6379,node3:6379",
    "redis-server-exe": "/usr/bin/redis-server"
}
server = ServerManager("redis_cluster", server_config)
server.start_server()

# Client side - use server info directly
datastore = DataStore("worker1", server.get_server_info())
```

4. Dragon Dictionary Workflow:
```python
# Server side
server_config = {
    "type": "dragon",
    "server-address": "node1:7777,node2:7777",
    "server-options": {"n_nodes": 2, "wait_for_keys": True}
}
server = ServerManager("dragon_server", server_config)
server.start_server()

# Client side - use server info directly
server_info = server.get_server_info()
datastore = DataStore("worker1", server_info)
# OR use serialized string for transmission
serialized = server.serialize()
datastore = DataStore("worker1", serialized)
```

5. Serialization for Remote Deployment:
```python
# Server side
server = ServerManager("my_server", config)
server.start_server()

# Serialize server info to base64 string
serialized_server = server.serialize()

# Send serialized_server to remote clients via network, file, etc.

# Client side (potentially on different machine)
datastore = DataStore("remote_client", serialized_server)
# DataStore automatically deserializes and configures client
```
"""

import time
import os
import sys
import pickle
import logging as logging_
import sqlite3
import shutil
import subprocess
import redis
import socket
from redis.cluster import RedisCluster
import zlib
from redis.cluster import ClusterNode
import base64
try:
    import dragon
    from dragon.data.ddict import DDict
    DRAGON_AVAILABLE=True
except:
    DRAGON_AVAILABLE=False

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    import pickle as cloudpickle
    CLOUDPICKLE_AVAILABLE = False
from typing import Union, List, Dict

class ServerManager:
    """
    Manages server setup and teardown for Redis and Dragon dictionary servers.
    Responsible for launching, monitoring, and stopping server processes.
    """
    def __init__(self, name, config: dict, logging=False, log_level=logging_.INFO):
        self.name = name
        self.config = config
        self.redis_processes = []  # Changed from single process to list
        self.dragon_dict = None
        
        # Setup logging
        if logging:
            self.logger = logging_.getLogger(f"{name}_server")
            self.logger.setLevel(log_level)
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"{name}_server.log")
            file_handler = logging_.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging_.INFO)
            
            formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.debug(f"ServerManager {name} initialized with config {config}")
        else:
            self.logger = None
    
    def start_server(self):
        self._setup_server()

    def _setup_server(self):
        """Setup the appropriate server based on configuration."""
        if self.logger:
            self.logger.info(f"Setting up {self.config['type']} server on {self.config.get('server-address', 'unknown')}")
        
        if self.config["type"] == "filesystem":
            if "server-address" not in self.config:
                self.config["server-address"] = os.path.join(os.getcwd(), ".tmp")
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            dirname = self.config["server-address"]
            os.makedirs(dirname, exist_ok=True)
            if self.logger:
                self.logger.info(f"Created filesystem directory at {dirname}")
        
        elif self.config["type"] == "node-local":
            self.config["server-address"] = "/tmp"
            if "nshards" not in self.config:
                self.config["nshards"] = 64
            if self.logger:
                self.logger.info(f"Using node-local directory {self.config['server-address']}")
        
        elif self.config["type"] == "redis":
            if "redis-server-exe" not in self.config:
                raise ValueError("redis-server-exe must be specified for Redis server")
            if "server-address" not in self.config:
                raise ValueError("Server address is required")
            self.redis_processes = self._start_redis_server()
        
        elif self.config["type"] == "dragon":
            if not DRAGON_AVAILABLE:
                raise ValueError("Dragon is not available")
            self.dragon_dict = self._start_dragon_dictionary()
            if self.dragon_dict:
                self.dragon_dict.setup_logging()
                if isinstance(self.dragon_dict, DDict):
                    if self.logger:
                        self.logger.info("Dragon dictionary created successfully!")
                else:
                    if self.logger:
                        self.logger.warning("Dragon dictionary creation failed!")
    
    def _start_redis_server(self):
        """Start Redis server processes for all addresses."""
        addresses = self.config["server-address"].split(",")
        is_clustered = self.config["db-type"] == "clustered"
        redis_processes = []
        
        for address in addresses:
            host = address.strip().split(":")[0]
            port = int(address.strip().split(":")[1])
            
            cmd_base = f"mpirun -np 1 -ppn 1 -hosts {host} {self.config.get('server-options',{}).get('mpi-options','')} {self.config['redis-server-exe']} --port {port} --bind 0.0.0.0 --protected-mode no"
            cmd = f"{cmd_base} --cluster-enabled yes --cluster-config-file {self.name}_{host}_{port}.conf" if is_clustered else cmd_base
                
            redis_process = subprocess.Popen(cmd, shell=True, env=os.environ.copy(), stdout=subprocess.DEVNULL)
            if self.logger:
                self.logger.debug(f"Started Redis {'cluster ' if is_clustered else ''}server at {address}")
            
            # Wait for Redis server to be ready
            self._wait_for_redis_server(host, port)
            redis_processes.append(redis_process)
        
        return redis_processes
    
    def _start_dragon_dictionary(self):
        """Start a Dragon dictionary server."""
        addresses = self.config["server-address"].split(",")
        nodes = [address.split(":")[0] for address in addresses]
        ports = [address.split(":")[1] for address in addresses]
        n_nodes = len(nodes)
        n_nodes_in = self.config.get("server-options", {}).get("n_nodes", None)
        
        if n_nodes_in is not None and n_nodes_in != n_nodes:
            if self.logger:
                self.logger.warning("Number of nodes in server-address differs from options. Using server-address count.")
            self.config["server-options"]["n_nodes"] = n_nodes
        
        policies = []
        for node in nodes:
            policy = dragon.infrastructure.policy.Policy(
                placement=dragon.infrastructure.policy.Policy.Placement.HOST_NAME,
                host_name=node
            )
            policies.append(policy)
        
        opts = {
            "n_nodes": n_nodes,
            "policy": policies
        }
        
        if "policy" in self.config.get("server-options", {}):
            if self.logger:
                self.logger.warning("Policy option provided as input. Replacing it!")
            self.config["server-options"]["policy"] = opts["policy"]
        
        opts.update(self.config.get("server-options", {}))
        opts["n_nodes"] = None
        opts["managers_per_node"] = None
        
        d = DDict(**opts)
        if self.logger:
            self.logger.info(f"Dragon dictionary created with options {opts}")
        return d
    
    def _wait_for_redis_server(self, host, port, max_retries=30, retry_delay=1.0):
        """Wait for Redis server to be ready by attempting connections."""
        if self.logger:
            self.logger.info(f"Waiting for Redis server at {host}:{port} to be ready...")
        
        for attempt in range(max_retries):
            try:
                sock = socket.create_connection((host, port), timeout=5)
                sock.close()
                
                test_client = redis.Redis(host=host, port=port, socket_connect_timeout=5)
                test_client.ping()
                test_client.close()
                
                if self.logger:
                    self.logger.info(f"Redis server at {host}:{port} is ready (attempt {attempt + 1})")
                return
                
            except (socket.timeout, socket.error, redis.ConnectionError, redis.TimeoutError) as e:
                if self.logger and attempt == 0:
                    self.logger.debug(f"Redis server not ready yet: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    error_msg = f"Redis server at {host}:{port} failed to start after {max_retries} attempts"
                    if self.logger:
                        self.logger.error(error_msg)
                    raise ConnectionError(error_msg)
    
    def poll_redis_server(self):
        """Check if all Redis server processes are still running."""
        if self.redis_processes:
            return all(process.poll() is None for process in self.redis_processes)
        return False
    
    def stop_server(self):
        """Stop the managed server."""
        if self.logger:
            self.logger.info("Stopping server!")
        
        if self.config["type"] == "redis" and self.redis_processes:
            for process in self.redis_processes:
                process.terminate()
                process.wait()
            if self.logger:
                self.logger.info(f"Stopped {len(self.redis_processes)} Redis server(s)")
        
        elif self.config["type"] == "dragon" and self.dragon_dict:
            self.dragon_dict.destroy()
            if self.logger:
                self.logger.info("Dragon dictionary destroyed")
        
        if self.logger:
            self.logger.info("Done stopping server!")
    
    def get_server_info(self):
        """Get information about the managed server."""
        info = {
            "name": self.name,
            "type": self.config["type"],
            "config": self.config.copy()  # Include the full config
        }
        
        if self.config["type"] == "redis":
            info["running"] = self.poll_redis_server()
        elif self.config["type"] == "dragon":
            info["dragon_dict"] = self.dragon_dict
            if self.dragon_dict and hasattr(self.dragon_dict, 'serialize'):
                info["serial_dragon_dict"] = self.dragon_dict.serialize()
        
        return info

    @classmethod
    def create_redis_cluster(cls, server_addresses: list, redis_cli_path: str = "redis-cli", 
                           replicas: int = 0, timeout: int = 30, logging=True):
        """
        Create a Redis cluster from existing Redis server instances.
        
        Args:
            server_addresses: List of Redis server addresses (e.g., ["host1:6379", "host2:6379"])
            redis_cli_path: Path to redis-cli executable (default: "redis-cli")
            replicas: Number of replicas per master (default: 0)
            timeout: Timeout in seconds for cluster creation (default: 30)
            logging: Whether to enable logging (default: True)
        
        Returns:
            True if cluster creation was successful, False otherwise
        
        Raises:
            ValueError: If server_addresses is empty or redis-cli command fails
        """
        if not server_addresses:
            raise ValueError("server_addresses cannot be empty")
        
        # Setup logging if enabled
        logger = None
        if logging:
            logger = logging_.getLogger("redis_cluster_creator")
            logger.setLevel(logging_.INFO)
            
            # Only add handler if not already present
            if not logger.handlers:
                console_handler = logging_.StreamHandler()
                formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        
        try:
            start_time = time.time()
            
            # Build the redis-cli cluster create command
            create_cmd = (
                f"{redis_cli_path} --cluster create "
                f"{' '.join(server_addresses)} "
                f"--cluster-replicas {replicas} "
                f"--cluster-yes"
            )
            
            if logger:
                logger.info(f"Creating Redis cluster with addresses: {server_addresses}")
                logger.debug(f"Executing command: {create_cmd}")
            
            # Execute the cluster creation command
            env = os.environ.copy()
            result = subprocess.run(
                create_cmd, 
                shell=True, 
                check=True, 
                env=env, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True
            )
            
            elapsed_time = time.time() - start_time
            
            if logger:
                logger.info(f"Redis cluster created successfully in {elapsed_time:.2f}s")
                if result.stdout:
                    logger.debug(f"Command stdout: {result.stdout}")
                if result.stderr:
                    logger.debug(f"Command stderr: {result.stderr}")
            
            return True
            
        except subprocess.TimeoutExpired:
            error_msg = f"Redis cluster creation timed out after {timeout} seconds"
            if logger:
                logger.error(error_msg)
            raise TimeoutError(error_msg)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Redis cluster creation failed with return code {e.returncode}"
            if logger:
                logger.error(error_msg)
                if e.stdout:
                    logger.error(f"Command stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"Command stderr: {e.stderr}")
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error during Redis cluster creation: {e}"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def serialize(self):
        """
        Serialize the server object for transmission or storage.
        
        Returns:
            str: Base64-encoded serialized server info
            
        Raises:
            RuntimeError: If serialization fails
        """
        try:
            server_info = self.get_server_info()
            serialized_bytes = cloudpickle.dumps(server_info)
            serialized_str = base64.b64encode(serialized_bytes).decode('utf-8')
            if self.logger:
                self.logger.info(f"Server info serialized successfully")
            return serialized_str
        except Exception as e:
            error_msg = f"Failed to serialize server info: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    @classmethod
    def deserialize(cls, serialized_data):
        """
        Deserialize server data and return server info.
        
        Args:
            serialized_data (str): Base64-encoded serialized server data
            
        Returns:
            dict: Server info dictionary
            
        Raises:
            RuntimeError: If deserialization fails
        """
        try:
            # Decode base64 string to bytes
            serialized_bytes = base64.b64decode(serialized_data.encode('utf-8'))
            
            # Use cloudpickle to deserialize
            server_info = cloudpickle.loads(serialized_bytes)
            
            if not isinstance(server_info, dict):
                raise ValueError("Invalid serialized server data format")
            
            return server_info
                
        except Exception as e:
            error_msg = f"Failed to deserialize server data: {e}"
            raise RuntimeError(error_msg)


class DataStore:
    """
    Handles client-side data operations including read, write, send, receive, and staging.
    Works with various backends: filesystem, node-local, Redis, and Dragon.
    
    Initialized with serialized server info from ServerManager.serialize() or ServerManager.get_server_info()
    """
    def __init__(self, name, server_info:Union[str, dict], logging=False, log_level=logging_.INFO):
        self.name = name
        self.connections = []
        self.redis_client = None
        self.dragon_dict = None

                # Setup logging
        if logging:
            self.logger = logging_.getLogger(f"{name}_datastore")
            self.logger.setLevel(log_level)
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
        
            log_file = os.path.join(log_dir, f"{name}_datastore.log")
            file_handler = logging_.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging_.INFO)
        
            formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            self.logger = None

        # Use ServerManager.deserialize to get server info and extract client config
        if isinstance(server_info, str):
            # Handle base64-encoded serialized data
            deserialized_server_info = ServerManager.deserialize(server_info)
            self.config = deserialized_server_info["config"].copy()
            # Add Dragon-specific info if needed
            if deserialized_server_info.get("type") == "dragon" and "serial_dragon_dict" in deserialized_server_info:
                self.config["server-obj"] = deserialized_server_info["serial_dragon_dict"]
        elif isinstance(server_info, dict):
            if "config" in server_info:
                self.config = server_info["config"].copy()
                # Add Dragon-specific info if needed
                if server_info.get("type") == "dragon":
                    if server_info.get("dragon_dict", None) is not None and isinstance(server_info["dragon_dict"], DDict):
                        if self.logger:
                            self.logger.info("Using provided Dragon dictionary object")
                        self.config["server-obj"] = server_info["dragon_dict"]
                    elif "serial_dragon_dict" in server_info:
                        self.config["server-obj"] = server_info["serial_dragon_dict"]
            else:
                raise ValueError("Invalid server info dict format")
        else:
            raise ValueError("server_info must be str (base64) or dict")

        if self.logger:
            self.logger.debug(f"DataStore {name} initialized with config {self.config}")
        # Initialize client connections
        self._setup_client()
    
    def _setup_client(self):
        """Setup client connections based on configuration."""
        if self.config["type"] == "filesystem":
            if "server-address" not in self.config:
                self.config["server-address"] = os.path.join(os.getcwd(), ".tmp")
            if "nshards" not in self.config:
                self.config["nshards"] = 64

        elif self.config["type"] == "node-local":
            if "server-address" not in self.config:
                self.config["server-address"] = "/tmp"
            if "nshards" not in self.config:
                self.config["nshards"] = 64

        elif self.config["type"] == "redis":
            if "server-address" not in self.config:
                raise ValueError("Server address is required for Redis client")
            self.redis_client = self._create_redis_client()
        
        elif self.config["type"] == "dragon":
            if not DRAGON_AVAILABLE:
                raise ValueError("Dragon is not available")
            if isinstance(self.config["server-obj"], bytes):
                self.dragon_dict = DDict.attach(self.config["server-obj"], trace=True)
            elif isinstance(self.config["server-obj"], str):
                self.dragon_dict = DDict.attach(self.config["server-obj"], trace=True)
            elif isinstance(self.config["server-obj"], DDict):
                self.dragon_dict = self.config["server-obj"]
            else:
                raise ValueError("Unknown server-obj type for Dragon client")
    
    def _create_redis_client(self):
        """Create a Redis client connection."""
        is_clustered = self.config["db-type"] == "clustered"
        clients = []
        
        try:
            if is_clustered:
                hosts = []
                ports = []
                for address in self.config["server-address"].split(","):
                    host, port_str = address.split(":")
                    port = int(port_str)
                    hosts.append(host)
                    ports.append(port)
                
                # Check which hosts are reachable
                reachable_hosts = []
                reachable_ports = []
                for host, port in zip(hosts, ports):
                    try:
                        sock = socket.create_connection((host, port), timeout=5)
                        sock.close()
                        reachable_hosts.append(host)
                        reachable_ports.append(port)
                        if self.logger:
                            self.logger.debug(f"Host {host}:{port} is reachable")
                    except (socket.timeout, socket.error) as e:
                        if self.logger:
                            self.logger.warning(f"Host {host}:{port} is not reachable: {e}")

                if not reachable_hosts:
                    error_msg = "No reachable Redis hosts found"
                    if self.logger:
                        self.logger.error(error_msg)
                    raise ConnectionError(error_msg)

                hosts = reachable_hosts
                ports = reachable_ports
                startup_nodes = [ClusterNode(host=host, port=port) for host, port in zip(hosts, ports)]
                client = RedisCluster(startup_nodes=startup_nodes)
                client.ping()
                clients.append(client)
                
            elif self.config["db-type"] == "colocated":
                my_hostname = socket.gethostname()
                for address in self.config["server-address"].split(","):
                    if my_hostname not in address:
                        if self.logger:
                            self.logger.warning(f"Skipping address {address} as it does not match hostname {my_hostname}")
                        continue
                    else:
                        if self.logger:
                            self.logger.info(f"Creating colocated Redis client for address {address}")
                    host, port_str = address.split(":")
                    port = int(port_str)
                    client = redis.Redis(host=host, port=port)
                    client.ping()
                    clients.append(client)
                assert len(clients) > 0, "No colocated Redis clients created"
                
            else:  # Non-clustered Redis server
                for address in self.config["server-address"].split(","):
                    host, port_str = address.split(":")
                    port = int(port_str)
                    client = redis.Redis(host=host, port=port)
                    client.ping()
                    clients.append(client)
            
            if self.logger:
                self.logger.debug(f"Connected to Redis {'cluster' if is_clustered else 'server'}")
            return clients
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def connect(self, other_node):
        """Connect this node to another node."""
        if other_node not in self.connections:
            self.connections.append(other_node)
            if self.logger:
                self.logger.debug(f"Connected to {other_node.name}")

    def disconnect(self, other_node):
        """Disconnect this node from another node."""
        if other_node in self.connections:
            self.connections.remove(other_node)
            if self.logger:
                self.logger.debug(f"Disconnected from {other_node.name}")

    def send(self, data, targets: list = None):
        """Send data to all or selected connections."""
        targets = targets or self.connections
        if self.logger:
            self.logger.debug(f"Sending data: {data}")
        
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
            os.makedirs(dirname, exist_ok=True)
            for target in targets:
                filename = os.path.join(dirname, f"{self.name}_{target.name}_data.pickle")
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
                if self.logger:
                    self.logger.debug(f"Data sent to {target.name} at {filename}")
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")

    def receive(self, senders: list = None):
        """Receive data from connected senders."""
        data = {}
        senders = senders or self.connections
        dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
        
        if not os.path.exists(dirname):
            if self.logger:
                self.logger.error(f"Directory {dirname} does not exist")
            raise AssertionError(f"Directory {dirname} does not exist")
            
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            for sender in senders:
                filename = os.path.join(dirname, f"{sender.name}_{self.name}_data.pickle")
                if not os.path.exists(filename):
                    if self.logger:
                        self.logger.error(f"File {filename} does not exist")
                    raise AssertionError(f"File {filename} does not exist")
                    
                with open(filename, "rb") as f:
                    data[sender.name] = pickle.load(f)
                    if self.logger:
                        self.logger.debug(f"Received data from {sender.name}")
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
        return data
    
    def stage_write(self, key, data, persistant: bool = True, client_id: int = 0):
        """Stage data as a key-value pair."""
        if self.config["type"] == "dragon":
            assert DRAGON_AVAILABLE, "dragon is not available"
            try:
                wait_for_keys = self.config.get("server-options", {}).get("wait_for_keys", None)
                if wait_for_keys is not None and wait_for_keys == True:
                    if persistant:
                        self.dragon_dict.pput(key, data)
                    else:
                        self.dragon_dict[key] = data
                else:
                    self.dragon_dict[key] = data
                    if self.logger and not persistant:
                        self.logger.warning("Doing a persistant put!")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Writing {key} failed with exception {e}")
                raise
                
        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            
            try:
                serialized_data = pickle.dumps(data)
                self.redis_client[client_id].set(key, serialized_data)
                if self.logger:
                    self.logger.debug(f"Staged data for {key} in Redis")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to stage data in Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config["server-address"]
            current_time = str(time.time())
            filename = os.path.join(dirname, f"{self.name}_{current_time}.pickle")

            with open(filename, "wb") as f:
                pickle.dump(data, f)
            
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            db_path = os.path.join(dirname, f"staging_{shard_number}.db")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS staging (
                    key TEXT PRIMARY KEY,
                    filename TEXT
                )
            """)

            max_retries = 5
            retry_delay = 0.5
            attempt = 0
            
            while attempt < max_retries:
                try:
                    cursor.execute("INSERT INTO staging (key, filename) VALUES (?, ?)", (key, filename))
                    conn.commit()
                    if self.logger:
                        self.logger.debug(f"Staged data for {key} at {filename} and recorded in database {db_path}")
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) or "readonly database" in str(e):
                        attempt += 1
                        if attempt < max_retries:
                            if self.logger:
                                self.logger.warning(f"Database {db_path} is locked/readonly. Waiting {retry_delay}s before retry {attempt}/{max_retries}")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                        else:
                            if self.logger:
                                self.logger.error(f"Failed to write to database after {max_retries} attempts: {e}")
                            raise
                    else:
                        if self.logger:
                            self.logger.error(f"Database error: {e}")
                        raise
                except sqlite3.IntegrityError:
                    if self.logger:
                        self.logger.error(f"Key {key} already exists in the staging database")
                    raise ValueError(f"Key {key} already exists")
            conn.close()
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
    
    def stage_read(self, key, client_id: int = 0):
        """Read staged data using the key."""
        if self.config["type"] == "dragon":
            try:
                return self.dragon_dict[key]
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Reading {key} failed with exception {e}")
                raise
                
        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            
            try:
                serialized_data = self.redis_client[client_id].get(key)
                if serialized_data is None:
                    if self.logger:
                        self.logger.error(f"Key {key} not found in Redis")
                    raise ValueError(f"Key {key} not found in Redis")
                    
                data = pickle.loads(serialized_data)
                if self.logger:
                    self.logger.debug(f"Read staged data for {key} from Redis")
                return data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to read data from Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            db_path = os.path.join(self.config["server-address"], f"staging_{shard_number}.db")

            if not os.path.exists(db_path):
                if self.logger:
                    self.logger.error(f"Database {db_path} does not exist")
                raise AssertionError(f"Database {db_path} does not exist")
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT filename FROM staging WHERE key=?", (key,))
            row = cursor.fetchone()
            conn.close()

            if row is None:
                if self.logger:
                    self.logger.error(f"Key {key} not found in the staging database")
                raise ValueError(f"Key {key} not found in the staging database")

            filename = row[0]
            if not os.path.exists(filename):
                if self.logger:
                    self.logger.error(f"File {filename} does not exist")
                raise AssertionError(f"File {filename} does not exist")
                
            with open(filename, "rb") as f:
                data = pickle.load(f)
                if self.logger:
                    self.logger.debug(f"Read staged data for {key} from {filename}")
                return data
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
    
    def poll_staged_data(self, key, client_id: int = 0):
        """Check if data for the key is staged."""
        if self.config["type"] == "dragon":
            try:
                if self.logger:
                    self.logger.debug(f"looking for {key} in dragon dict keys {self.dragon_dict.keys()}")
                return key in self.dragon_dict.keys()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Polling {key} failed with exception {e}")
                return False
                
        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            
            try:
                return self.redis_client[client_id].exists(key) > 0
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to poll data in Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            db_path = os.path.join(self.config["server-address"], f"staging_{shard_number}.db")

            if not os.path.exists(db_path):
                return False
                
            if self.logger:
                self.logger.debug(f"Polling for {key} in database {db_path}")
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            try:
                cursor.execute("SELECT filename FROM staging WHERE key=?", (key,))
                row = cursor.fetchone()
                conn.close()
            except sqlite3.OperationalError as e:
                return False

            return row is not None
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
        
    def clean_staged_data(self, key, client_id: int = 0):
        """Clear the staging area for the given key."""
        if self.config["type"] == "dragon":
            try:
                self.dragon_dict.pop(key)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Deleting {key} failed with exception {e}")
                    
        elif self.config["type"] == "redis":
            if not self.redis_client:
                raise ValueError("Redis client not initialized")
            
            try:
                if not self.redis_client[client_id].exists(key):
                    if self.logger:
                        self.logger.error(f"Key {key} not found in Redis")
                    raise ValueError(f"Key {key} not found in Redis")
                
                self.redis_client[client_id].delete(key)
                if self.logger:
                    self.logger.debug(f"Cleared staged data for {key} from Redis")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to clean data in Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            db_path = os.path.join(self.config["server-address"], f"staging_{shard_number}.db")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT filename FROM staging WHERE key=?", (key,))
            row = cursor.fetchone()

            if row is None:
                if self.logger:
                    self.logger.error(f"Key {key} not found in the staging database")
                raise ValueError(f"Key {key} not found in the staging database")

            filename = row[0]
            max_retries = 5
            retry_delay = 0.5
            attempt = 0
            
            while attempt < max_retries:
                try:
                    cursor.execute("DELETE FROM staging WHERE key=?", (key,))
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) or "readonly database" in str(e):
                        attempt += 1
                        if attempt < max_retries:
                            if self.logger:
                                self.logger.warning(f"Database {db_path} is locked/readonly. Waiting {retry_delay}s before retry {attempt}/{max_retries}")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                        else:
                            if self.logger:
                                self.logger.error(f"Failed to delete from database after {max_retries} attempts: {e}")
                            raise
                    else:
                        if self.logger:
                            self.logger.error(f"Database error: {e}")
                        raise
            conn.close()

            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    if self.logger:
                        self.logger.debug(f"Cleared staged data for {key} and deleted file {filename}")
                except:
                    if self.logger:
                        self.logger.warning(f"Failed to delete file {filename}. The file may be in use or you may not have permission.")
            else:
                if self.logger:
                    self.logger.error(f"File {filename} does not exist")
                raise ValueError(f"File {filename} does not exist")
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")

    def get_connections(self):
        """Return a list of connected nodes."""
        return self.connections

    def clean(self):
        """Clean up the datastore."""
        if self.config["type"] == "filesystem":
            dirname = self.config.get("server-address", os.path.join(os.getcwd(), ".tmp"))
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
                if self.logger:
                    self.logger.debug(f"Cleaned up directory {dirname}")
        elif self.config["type"] == "node-local":
            fname = os.path.join(self.config["server-address"], "staging.db")
            if os.path.exists(fname):
                os.remove(fname)
        elif self.config["type"] == "dragon":
            if self.dragon_dict:
                self.dragon_dict.detach()
                if self.logger:
                    self.logger.info("Dragon client detached")
        else:
            if self.logger:
                self.logger.debug("No cleanup needed for this backend type")

    def flush_logger(self):
        if self.logger:
            for handler in self.logger.handlers:
                handler.flush()

    def __repr__(self):
        return f"<DataStore name={self.name}, type={self.config['type']}>"

