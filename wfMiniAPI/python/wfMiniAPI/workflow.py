import subprocess
import os
import sys
import time
import argparse
import json
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from wfMiniAPI.component import Component
from wfMiniAPI.launcher import BasicLauncher


@dataclass
class WorkflowComponent:
    """Represents a component in the workflow."""
    # Required fields (no defaults) must come first
    name: str
    executable: Union[str, Callable]
    config: Dict[str, Any]
    type: str = "local"  # should belong to the list ["local", "remote", "dragon"]
    nodes: List[str] = field(default_factory=list)
    ppn: int = 1
    cpu_affinity: List[int] = field(default_factory=list)
    gpu_affinity: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Other component names this depends on


@dataclass
class ServerComponent:
    """Represents a server component in the workflow (e.g., database, web server)."""
    name: str
    server_type: str  # e.g., 'redis', 'dragon', 'http', 'tcp'
    host: str = "localhost"
    port: int = 8080
    config: Dict[str, Any] = field(default_factory=dict)
    nodes: List[str] = field(default_factory=list)
    startup_command: Optional[str] = None  # Command to start the server
    health_check_url: Optional[str] = None  # URL to check if server is healthy
    startup_timeout: int = 30  # Seconds to wait for server startup
    dependencies: List[str] = field(default_factory=list)  # Other components this depends on
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        """Get the server address as host:port."""
        return f"{self.host}:{self.port}"
    
    def is_local(self) -> bool:
        """Check if this is a local server."""
        return self.host in ["localhost", "127.0.0.1", "0.0.0.0"]


class Workflow:
    """
    Generic workflow orchestration class responsible for registering components 
    and coordinating their execution via a launcher.
    """
    
    def __init__(self, **config_files):
        """
        Initialize workflow.
        
        Args:
            **config_files: Named configuration files (e.g., ai_config="path.json")
        """
        # Store configuration files and loaded configs
        self.config_files = {key: value for key, value in config_files.items() if isinstance(value, str)}
        self.configs = {key: value for key, value in config_files.items() if isinstance(value, dict)}
        
        # Load configurations if provided
        for config_name, config_path in self.config_files.items():
            if config_path:
                with open(config_path, 'r') as f:
                    self.configs[config_name] = json.load(f)
        
        # Registered workflow components
        self.components: Dict[str, WorkflowComponent] = {}
        
        # Registered server components
        self.servers: Dict[str, ServerComponent] = {}
        
        # Launcher instance
        self.launcher = BasicLauncher(launcher_config=self.configs.get('launcher', {"mode": "mpi"}))
    
    def register_component(self, name: str, 
                          executable: Union[str, Callable], 
                          config: Dict[str, Any] = None, 
                          nodes: List[str] = None,
                          ppn: int = 1, 
                          cpu_affinity: List[int] = None,
                          gpu_affinity: List[str] = None, 
                          env_vars: Dict[str, str] = None,
                          dependencies: List[str] = None) -> 'Workflow':
        """
        Register a component in the workflow.
        
        Args:
            name: Unique name for this component
            executable: Command string or Python function to execute
            config: Configuration dictionary for the component
            nodes: List of nodes to run on (optional)
            ppn: Processes per node (optional)
            cpu_affinity: CPU cores to bind to (optional)
            gpu_affinity: GPU devices to bind to (optional)
            env_vars: Environment variables (optional)
            dependencies: List of component names this depends on (optional)
            
        Returns:
            Self for method chaining
        """
        component = WorkflowComponent(
            name=name,
            executable=executable,
            config=config or {},
            nodes=nodes or [],
            ppn=ppn,
            cpu_affinity=cpu_affinity or [],
            gpu_affinity=gpu_affinity or [],
            env_vars=env_vars or {},
            dependencies=dependencies or []
        )
        
        self.components[name] = component
        return self
    
    def register_server(self, name: str,
                       server_type: str,
                       host: str = "localhost",
                       port: int = 8080,
                       config: Dict[str, Any] = None,
                       nodes: List[str] = None,
                       startup_command: str = None,
                       health_check_url: str = None,
                       startup_timeout: int = 30,
                       dependencies: List[str] = None,
                       env_vars: Dict[str, str] = None) -> 'Workflow':
        """
        Register a server component in the workflow.
        
        Args:
            name: Unique name for this server
            server_type: Type of server ('redis', 'dragon', 'http', 'tcp', etc.)
            host: Host address (default: localhost)
            port: Port number (default: 8080)
            config: Configuration dictionary for the server
            nodes: List of nodes to run on (optional)
            startup_command: Command to start the server (optional)
            health_check_url: URL to check server health (optional)
            startup_timeout: Seconds to wait for startup (default: 30)
            dependencies: List of component names this depends on (optional)
            env_vars: Environment variables (optional)
            
        Returns:
            Self for method chaining
        """
        server = ServerComponent(
            name=name,
            server_type=server_type,
            host=host,
            port=port,
            config=config or {},
            nodes=nodes or [],
            startup_command=startup_command,
            health_check_url=health_check_url,
            startup_timeout=startup_timeout,
            dependencies=dependencies or [],
            env_vars=env_vars or {}
        )
        
        self.servers[name] = server
        return self

    def get_component(self, name: str) -> Optional[WorkflowComponent]:
        """Get a registered component by name."""
        return self.components.get(name)
    
    def get_server(self, name: str) -> Optional[ServerComponent]:
        """Get a registered server by name."""
        return self.servers.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self.components.keys())
    
    def list_servers(self) -> List[str]:
        """List all registered server names."""
        return list(self.servers.keys())

    def _resolve_execution_order(self) -> List[List[str]]:
        """
        Resolve the execution order of components based on dependencies.
        
        Returns:
            List of component groups that can be executed in parallel
        """
        # Simple topological sort for dependency resolution
        remaining = set(self.components.keys())
        execution_order = []
        
        while remaining:
            # Find components with no unresolved dependencies
            ready = []
            for comp_name in remaining:
                comp = self.components[comp_name]
                if all(dep not in remaining for dep in comp.dependencies):
                    ready.append(comp_name)
            
            if not ready:
                # Circular dependency or missing dependency
                raise ValueError(f"Circular dependency detected or missing components: {remaining}")
            
            execution_order.append(ready)
            remaining -= set(ready)
        
        return execution_order
    
    def _launch_component(self, component: WorkflowComponent):
        """
        Launch a single workflow component.
        
        Args:
            component: The component to launch
            
        Returns:
            Process object or result code
        """
        return self.launcher.launch_component(component)
    
    def launch(self, **kwargs) -> int:
        """
        Execute the complete workflow by launching all registered components
        in dependency order.
        
        Args:
            **kwargs: Additional arguments passed to component handlers
        
        Returns:
            0 for success, 1 for failure
        """
        if not self.components:
            print("No components registered in workflow")
            return 0
            
        try:
            # Get execution order based on dependencies
            execution_order = self._resolve_execution_order()
            
            # Execute components in dependency order
            for component_group in execution_order:
                group_processes = []
                
                for component_name in component_group:
                    component = self.components[component_name]
                    print(f"Launching component: {component_name}")
                    
                    # Launch the component
                    process_or_result = self._launch_component(component)
                    group_processes.append((component_name, process_or_result))
                
                # Wait for this group to complete before starting the next
                if group_processes:
                    for component_name, process_or_result in group_processes:
                        return_code = self.launcher.wait_for_component(process_or_result)
                        
                        if return_code == 0:
                            print(f"Component {component_name} completed successfully")
                        else:
                            print(f"Component {component_name} failed with return code {return_code}")
                            return 1  # Fail fast on first error
            
            print("All workflow components completed successfully")
            return 0
            
        except Exception as e:
            print(f"Workflow execution failed: {e}")
            return 1

    def component(self, func: Callable = None, *, 
                 name: str = None,
                 config: Dict[str, Any] = None, 
                 nodes: List[str] = None,
                 ppn: int = 1, 
                 cpu_affinity: List[int] = None,
                 gpu_affinity: List[str] = None, 
                 env_vars: Dict[str, str] = None,
                 dependencies: List[str] = None):
        """
        Decorator to register a component in the workflow.
        
        Can be used with or without parentheses:
        - @workflow.component
        - @workflow.component()
        - @workflow.component(name="my_task", dependencies=["setup"])
        
        Args:
            func: Function to register (when used without parentheses)
            name: Unique name for this component (defaults to function name)
            config: Configuration dictionary for the component
            nodes: List of nodes to run on (optional)
            ppn: Processes per node (optional)
            cpu_affinity: CPU cores to bind to (optional)
            gpu_affinity: GPU devices to bind to (optional)
            env_vars: Environment variables (optional)
            dependencies: List of component names this depends on (optional)
            
        Returns:
            Decorated function or decorator function
            
        Examples:
            @workflow.component
            def my_function():
                return 0
                
            @workflow.component()
            def another_function():
                return 0
                
            @workflow.component(name="my_task", dependencies=["setup"])
            def third_function():
                return 0
        """
        def decorator(f: Callable):
            component_name = name if name is not None else f.__name__
            self.register_component(
                name=component_name,
                executable=f,
                config=config,
                nodes=nodes,
                ppn=ppn,
                cpu_affinity=cpu_affinity,
                gpu_affinity=gpu_affinity,
                env_vars=env_vars,
                dependencies=dependencies
            )
            return f
        
        # If func is provided, this was called without parentheses: @workflow.component
        if func is not None:
            return decorator(func)
        
        # Otherwise, this was called with parentheses: @workflow.component() or @workflow.component(args)
        return decorator

def main(config_files: Dict[str, str], **kwargs):
    """
    Main function to execute a generic workflow.
    
    Args:
        config_files: Dictionary of configuration files by name
        **kwargs: Additional arguments for workflow execution
    
    Returns:
        0 for success, 1 for failure
    """
    workflow = Workflow(**config_files)
    return workflow.launch(**kwargs)