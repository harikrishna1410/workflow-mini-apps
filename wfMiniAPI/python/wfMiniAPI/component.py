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
try:
    import dragon
    from dragon.data.ddict import DDict
    DRAGON_AVAILABLE=True
except:
    DRAGON_AVAILABLE=False

class Component:
    def __init__(self, name, config:dict={"type":"filesystem"},logging=False,log_level=logging_.INFO):
        self.name = name
        self.config = config
        self.connections = []
        if logging:
            # Setup logging
            self.logger = logging_.getLogger(name)
            self.logger.setLevel(log_level)
            # Create logs directory if it doesn't exist
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
        
            # Create file handler
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = logging_.FileHandler(log_file,mode="w")
            file_handler.setLevel(logging_.INFO)
        
            # Create formatter
            formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
        
            # Add handler to logger
            self.logger.addHandler(file_handler)
        
            self.logger.debug(f"Component {name} initialized with config {config}")
        else:
            self.logger = None
        
        self.redis_process = None
        self.redis_client = None
        self.dragon_dict = None


        """
        Example redis config:
        {
        "name" = "redis",
        "role" = "server or client"
        "db-type" = "clustered or colocated"
        "server-address" = "nodename:port",
        "server-options" = {"mpi-options":""} (optional)
        }
        """
        """
            Example dragon config:
            {
            "name" = "dragon",
            "role" = "server or client"
            "server-address" = "nodename1:port,nodename2:port",
            "server-obj" = serialize ddict string  obtained from ddict.serialize() (required for client role)
            "server-options" = {key:value} (optional,same as ddict options). provide  same server-options for client and server
            }
            """
        match self.config["type"]:
            case "filesystem":
                if "location" not in self.config:
                    self.config["location"] = os.path.join(os.getcwd(), ".tmp")
                if "nshards" not in config:
                    self.config["nshards"] = 64
                dirname = self.config.get("location")
                os.makedirs(dirname, exist_ok=True)

            case "node-local":
                self.config["location"] = "/tmp"
                if "nshards" not in config:
                    self.config["nshards"] = 64

            case "redis":
                if self.config["db-type"] not in ["clustered", "colocated"]:
                    raise ValueError(f"Unknown db type: {self.config['db-type']}")
                
                match self.config["role"]:
                    case "server":
                        if "redis-server-exe" not in self.config:
                            raise ValueError("redis-server-exe must be specified for server role")
                        if "server-address" not in self.config:
                            raise ValueError(f"Server address is required")
                        self.redis_process = self._start_redis_server()
                    
                    case "client":
                        if "server-address" not in self.config:
                            raise ValueError(f"Server address is required")
                        self.redis_client = self._create_redis_client()
                    
                    case "both":
                        if "redis-server-exe" not in self.config:
                            raise ValueError("redis-server-exe must be specified for 'both' role")
                        if "server-address" not in self.config:
                            raise ValueError(f"Server address is required")
                        self.redis_process = self._start_redis_server()
                        time.sleep(2)  # Give the server time to start up
                        self.redis_client = self._create_redis_client()
                    
                    case _:
                        raise ValueError(f"Unknown role for component {self.name}: {self.config['role']}")
            case "dragon":
                assert DRAGON_AVAILABLE, "Dragon is not available"
                match self.config["role"]:
                    ####Here, server implies manager in dragon.
                    case "server":
                        self.dragon_dict = self._start_dragon_dictionary()
                        self.dragon_dict.setup_logging()
                        if isinstance(self.dragon_dict,DDict):
                            if self.logger:
                                self.logger.info("ddcit creating successful!")
                        else:
                            if self.logger:
                                self.logger.warning("ddcit creating failed!")

                    case "client":
                        if isinstance(self.config["server-obj"],str):
                            self.dragon_dict = DDict.attach(self.config["server-obj"],trace=True)
                        elif isinstance(self.config["server-obj"],DDict):
                            self.dragon_dict = self.config["server-obj"]
                        else:
                            raise ValueError("Unknown server-obj")
                    case _:
                        raise ValueError(f"Unknown role {self.config['role']}")
            case _:
                raise ValueError(f"Unknown data transfer backend {self.config['type']}. "+\
                                 f"Supported backends {','.join(['filesystem','node-local','redis','dragon'])}")
        
        
    def _start_dragon_dictionary(self):
        addresses = self.config["server-address"].split(",")
        nodes = [address.split(":")[0] for address in addresses]
        ports = [address.split(":")[1] for address in addresses]
        n_nodes = len(nodes)
        n_nodes_in = self.config.get("server-options",{}).get("n_nodes",None)
        if n_nodes_in is not None and n_nodes_in != n_nodes:
            if self.logger:
                self.logger.warning(f"Number of nodes in server-address is not same as number of input nodes in options")
                self.logger.warning("Using the one from server-address")
            self.config["server-options"]["n_nodes"] = n_nodes
        policies = []
        for node in nodes:
            policy = dragon.infrastructure.policy.Policy(
                        placement=dragon.infrastructure.policy.Policy.Placement.HOST_NAME,
                        host_name=node
                    )
            policies.append(policy)
        opts = {
                "n_nodes":n_nodes,
                "policy":policies
                }
        if "policy" in self.config.get("server-options",{}):
            if self.logger:
                self.logger.warning(f"Policy option is give as an input.Replacing it!")
            self.config["server-options"]["policy"] = opts["policy"]
        opts.update(self.config.get("server-options",{}))
        opts["n_nodes"] = None
        opts["managers_per_node"] = None
        d = DDict(**opts)
        return d
        
    def _start_redis_server(self):
        """Start a Redis server process."""
        address = self.config["server-address"]
        is_clustered = self.config["db-type"] == "clustered"
        
        host = address.split(":")[0]
        port = address.split(":")[1]
        
        cmd_base = f"mpirun -np 1 -ppn 1 -hosts {host} {self.config.get('server-options',{}).get('mpi-options','')} {self.config['redis-server-exe']} --port {port} --bind 0.0.0.0 --protected-mode no"
        cmd = f"{cmd_base} --cluster-enabled yes --cluster-config-file {self.name}.conf" if is_clustered else cmd_base
            
        redis_process = subprocess.Popen(cmd, shell=True, env=os.environ.copy(),stdout=subprocess.DEVNULL)
        if self.logger:
            self.logger.debug(f"Started Redis {'cluster ' if is_clustered else ''}server at {address}")
        return redis_process

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
                    ##
                    hosts.append(host)
                    ports.append(port)
                # Check which hosts are reachable
                reachable_hosts = []
                reachable_ports = []
                for host, port in zip(hosts, ports):
                    try:
                        # Try to establish a connection to check if host is reachable
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

                # Update hosts and ports to only include reachable ones
                hosts = reachable_hosts
                ports = reachable_ports
                # Create a list of startup nodes for the Redis Cluster
                startup_nodes = [{"host": host, "port": port} for host, port in zip(hosts, ports)]
                startup_nodes = [ClusterNode(host=host, port=port) for host, port in zip(hosts, ports)]
                client = RedisCluster(startup_nodes=startup_nodes)
                client.ping()  # Test connection
                clients.append(client)
            else:
                for address in self.config["server-address"].split(","):
                    host, port_str = address.split(":")
                    port = int(port_str)
                    client = redis.Redis(host=host, port=port)
                    client.ping()  # Test connection
                    clients.append(client)
            if self.logger:
                self.logger.debug(f"Connected to Redis {'cluster' if is_clustered else 'server'} at {address}")
            return clients
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to connect to Redis {'cluster' if is_clustered else 'server'}: {e}")
            raise

    def connect(self, other_node:any):
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

    def send(self, data, targets:list=None):
        """
        Send data to all or selected connections.
        :param data: The data to send.
        """
        targets = targets or self.connections
        if self.logger:
            self.logger.debug(f"Sending data: {data}")
        
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            dirname = self.config.get("location", os.path.join(os.getcwd(), ".tmp"))
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

    def receive(self, senders:list=None):
        """
        Handle received data. Override this in subclasses for custom behavior.
        :param data: The data received.
        :param sender: The node that sent the data.
        """
        data = {}
        senders = senders or self.connections
        dirname = self.config.get("location", os.path.join(os.getcwd(), ".tmp"))
        
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
    
    def stage_write(self, key, data, persistant:bool=True, client_id:int=0):
        """
        Function stages data as a key-value pair.
        The key and filename are stored in a database, while the data is saved in a file.
        For Redis, the data is stored directly with the key.
        """

        if self.config["type"] == "dragon":
            assert DRAGON_AVAILABLE, "dragon is not available"
            try:
                wait_for_keys = self.config.get("server-options",{}).get("wait_for_keys",None)
                if wait_for_keys is not None and wait_for_keys == True:
                    if persistant:
                        self.dragon_dict.pput(key,data)
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
                # Serialize and store data directly in Redis
                serialized_data = pickle.dumps(data)
                self.redis_client[client_id].set(key, serialized_data)
                if self.logger:
                    self.logger.debug(f"Staged data for {key} in Redis")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to stage data in Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Ensure the directory for files exists
            dirname = self.config["location"]

            # Generate a unique filename for the data file
            current_time = str(time.time())
            filename = os.path.join(dirname, f"{self.name}_{current_time}.pickle")

            # Save the data to the file
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            # Path to the SQLite database
            db_path = os.path.join(dirname, f"staging_{shard_number}.db")

            # Connect to the database and store the key-filename mapping
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create the table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS staging (
                    key TEXT PRIMARY KEY,
                    filename TEXT
                )
            """)

            # Insert the key-filename mapping with retry mechanism for locked DB
            max_retries = 5
            retry_delay = 0.5  # seconds
            attempt = 0
            
            while attempt < max_retries:
                try:
                    cursor.execute("INSERT INTO staging (key, filename) VALUES (?, ?)", (key, filename))
                    conn.commit()
                    if self.logger:
                        self.logger.debug(f"Staged data for {key} at {filename} and recorded in database {db_path}")
                    break  # Success, exit the loop
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) or "readonly database" in str(e):
                        attempt += 1
                        if attempt < max_retries:
                            if self.logger:
                                self.logger.warning(f"Database {db_path} is locked/readonly. Waiting {retry_delay}s before retry {attempt}/{max_retries}")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
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
    
    def stage_read(self, key, client_id:int=0):
        """
            Function reads the data from a staging area using the key.
            For filesystem/node-local, the key is used to look up the filename in the database.
            For Redis, the data is retrieved directly using the key.
        """
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
                # Get data directly from Redis
                serialized_data = self.redis_client[client_id].get(key)
                if serialized_data is None:
                    if self.logger:
                        self.logger.error(f"Key {key} not found in Redis")
                    raise ValueError(f"Key {key} not found in Redis")
                    
                # Deserialize the data
                data = pickle.loads(serialized_data)
                if self.logger:
                    self.logger.debug(f"Read staged data for {key} from Redis")
                return data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to read data from Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Path to the SQLite database
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            db_path = os.path.join(self.config["location"], f"staging_{shard_number}.db")

            if not os.path.exists(db_path):
                if self.logger:
                    self.logger.error(f"Database {db_path} does not exist")
                raise AssertionError(f"Database {db_path} does not exist")
            # Connect to the database and retrieve the filename for the key
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query the database for the filename associated with the key
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
            # Load the data from the file
            with open(filename, "rb") as f:
                data = pickle.load(f)
                if self.logger:
                    self.logger.debug(f"Read staged data for {key} from {filename}")
                return data
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
    
    def poll_staged_data(self, key, client_id:int=0):
        """
        Function checks if the data for the key is staged.
        It returns True if the data is staged, otherwise False.
        """
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
                # Check if key exists in Redis
                return self.redis_client[client_id].exists(key) > 0
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to poll data in Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Path to the SQLite database
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            db_path = os.path.join(self.config["location"], f"staging_{shard_number}.db")

            if not os.path.exists(db_path):
                return False
            # Connect to the database and check if the key exists
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Query the database for the filename associated with the key
            cursor.execute("SELECT filename FROM staging WHERE key=?", (key,))
            row = cursor.fetchone()
            conn.close()

            if row is None:
                return False
            else:
                return True
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
        
    def clean_staged_data(self, key, client_id:int=0):
        """
        Function clears the staging area for the given key.
        For filesystem/node-local, it removes the key-filename mapping and deletes the file.
        For Redis, it deletes the key from the Redis database.
        """
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
                # Check if key exists
                if not self.redis_client[client_id].exists(key):
                    if self.logger:
                        self.logger.error(f"Key {key} not found in Redis")
                    raise ValueError(f"Key {key} not found in Redis")
                
                # Delete the key from Redis
                self.redis_client[client_id].delete(key)
                if self.logger:
                    self.logger.debug(f"Cleared staged data for {key} from Redis")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to clean data in Redis: {e}")
                raise
            
        elif self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Path to the SQLite database
            h = zlib.crc32(key.encode('utf-8'))
            shard_number = h % self.config["nshards"]
            db_path = os.path.join(self.config["location"], f"staging_{shard_number}.db")

            # Connect to the database and delete the key-filename mapping
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Query the database for the filename associated with the key
            cursor.execute("SELECT filename FROM staging WHERE key=?", (key,))
            row = cursor.fetchone()

            if row is None:
                if self.logger:
                    self.logger.error(f"Key {key} not found in the staging database")
                raise ValueError(f"Key {key} not found in the staging database")

            filename = row[0]
            # Delete the key-filename mapping from the database with retry mechanism
            max_retries = 5
            retry_delay = 0.5  # seconds
            attempt = 0
            
            while attempt < max_retries:
                try:
                    cursor.execute("DELETE FROM staging WHERE key=?", (key,))
                    conn.commit()
                    break  # Success, exit the loop
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) or "readonly database" in str(e):
                        attempt += 1
                        if attempt < max_retries:
                            if self.logger:
                                self.logger.warning(f"Database {db_path} is locked/readonly. Waiting {retry_delay}s before retry {attempt}/{max_retries}")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                        else:
                            if self.logger:
                                self.logger.error(f"Failed to delete from database after {max_retries} attempts: {e}")
                            raise
                    else:
                        if self.logger:
                            self.logger.error(f"Database error: {e}")
                        raise
            conn.close()

            # Delete the file
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

    def __repr__(self):
        return f"<WorkflowNode name={self.name}>"
    
    def clean(self):
        """Clean up the component."""
        if self.config["type"] == "filesystem":
            dirname = self.config.get("location", os.path.join(os.getcwd(), ".tmp"))
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
                if self.logger:
                    self.logger.debug(f"Cleaned up directory {dirname}")
        elif self.config["type"] == "node-local":
            fname = os.path.join(self.config["location"],"staging.db")
            if os.path.exists(fname):
                os.remove(fname)
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")

    def poll_redis_server(self):
        if self.redis_process:
            return self.redis_process.poll() is None
        return False
    
    def stop_redis_server(self):
        if self.redis_process:
            self.redis_process.terminate()
            self.redis_process.wait()
    
    def stop_server(self):
        if self.logger:
            self.logger.info("stopping server!")
        if self.config["type"] == "redis":
            self.stop_redis_server()
        elif self.config["type"] == "dragon":
            if self.dragon_dict and self.config["role"] == "server":
                self.dragon_dict.destroy()
        if self.logger:
            self.logger.info("sone stopping server!")
    
    def stop_client(self):
        if self.logger:
            self.logger.info("stopping client!")
        if self.config["type"] == "dragon":
            self.dragon_dict.detach()
        if self.logger:
            self.logger.info("done stopping client!")

