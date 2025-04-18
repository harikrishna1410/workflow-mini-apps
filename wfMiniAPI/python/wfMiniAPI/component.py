import time
class Component:
    def __init__(self, name):
        self.name = name
        self.connections = []

    def connect(self, other_node):
        """Connect this node to another node."""
        if other_node not in self.connections:
            self.connections.append(other_node)

    def disconnect(self, other_node):
        """Disconnect this node from another node."""
        if other_node in self.connections:
            self.connections.remove(other_node)

    def send(self, data, targets=None):
        """
        Send data to all or selected connections.
        :param data: The data to send.
        :param targets: Optional list of target nodes. If None, send to all connections.
        """
        targets = targets or self.connections
        print(f"{self.name} sending data: {data}")
        time.sleep(0.5)

    def receive(self, sender=None):
        """
        Handle received data. Override this in subclasses for custom behavior.
        :param data: The data received.
        :param sender: The node that sent the data.
        """
        data = "sample data"
        print(f"{self.name} received data from {sender.name if sender else 'unknown'}: {data}")
        time.sleep(0.5)
        return data

    def get_connections(self):
        """Return a list of connected nodes."""
        return self.connections

    def __repr__(self):
        return f"<WorkflowNode name={self.name}>"