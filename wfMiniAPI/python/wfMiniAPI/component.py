import time
import os
import pickle
import logging as logging_
import sqlite3
import shutil

class Component:
    def __init__(self, name, config:dict={"type":"filesystem"},logging=False):
        self.name = name
        self.config = config
        self.connections = []
        assert self.config["type"] in ["filesystem","node-local"]
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Ensure that tmp directory is empty
            dirname = self.config.get("location", os.path.join(os.getcwd(), ".tmp"))
            os.makedirs(dirname, exist_ok=True)
        
        if self.config["type"] == "node-local":
            self.config["location"] = "/tmp"
        
        if logging:
            # Setup logging
            self.logger = logging_.getLogger(name)
            self.logger.setLevel(logging_.INFO)
            # Create logs directory if it doesn't exist
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
        
            # Create file handler
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = logging_.FileHandler(log_file)
            file_handler.setLevel(logging_.INFO)
        
            # Create formatter
            formatter = logging_.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
        
            # Add handler to logger
            self.logger.addHandler(file_handler)
        
            self.logger.debug(f"Component {name} initialized with config {config}")
        else:
            self.logger = None

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
    
    def stage_write(self, key, data):
        """
        Function stages data as a key-value pair.
        The key and filename are stored in a database, while the data is saved in a file.
        """
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Ensure the directory for files exists
            dirname = self.config.get("location", os.path.join(os.getcwd(), ".tmp"))
            os.makedirs(dirname, exist_ok=True)

            # Generate a unique filename for the data file
            current_time = str(time.time())
            filename = os.path.join(dirname, f"{self.name}_{current_time}.pickle")

            # Save the data to the file
            with open(filename, "wb") as f:
                pickle.dump(data, f)

            # Path to the SQLite database
            db_path = os.path.join(dirname, f"staging.db")

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
                finally:
                    conn.close()
        else:
            if self.logger:
                self.logger.error("Unsupported data transport type")
            raise ValueError("Unsupported data transport type")
    
    def stage_read(self, key):
        """
            Function reads the data from a staging area using the key.
            the key is used to look up the filename in the database.
            The data is then loaded from the file.
        """
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Path to the SQLite database
            db_path = os.path.join(self.config.get("location", os.path.join(os.getcwd(), ".tmp")), f"staging.db")

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
    
    def poll_staged_data(self,key):
        """
        Function checks if the data for the key is staged.
        It returns True if the data is staged, otherwise False.
        """
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Path to the SQLite database
            db_path = os.path.join(self.config.get("location", os.path.join(os.getcwd(), ".tmp")), f"staging.db")

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
        
    def clean_staged_data(self,key):
        """
        Function clears the staging area for the given key.
        It removes the key-filename mapping from the database and deletes the file.
        """
        if self.config["type"] == "filesystem" or self.config["type"] == "node-local":
            # Path to the SQLite database
            db_path = os.path.join(self.config.get("location", os.path.join(os.getcwd(), ".tmp")), f"staging.db")

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

            # Delete the key-filename mapping from the database
            cursor.execute("DELETE FROM staging WHERE key=?", (key,))
            conn.commit()
            conn.close()

            # Delete the file
            if os.path.exists(filename):
                os.remove(filename)
                if self.logger:
                    self.logger.debug(f"Cleared staged data for {key} and deleted file {filename}")
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