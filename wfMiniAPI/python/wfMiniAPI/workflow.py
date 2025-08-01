import subprocess
import os
import sys
import time
import argparse
import json
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from wfMiniAPI.launcher import BasicLauncher


@dataclass
class WorkflowComponent:
    """Represents a component in the workflow."""
    # Required fields (no defaults) must come first
    name: str
    executable: Union[str, Callable]
    type: str  # should belong to the list ["local", "remote", "dragon"]
    args: Dict[str, Any] = field(default_factory=dict)  # Arguments for the component
    nodes: List[str] = field(default_factory=list)
    ppn: int = 1
    num_gpus_per_process: int = 0
    cpu_affinity: List[int] = None
    gpu_affinity: List[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Other component names this depends on


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
        
        # Launcher instance
        self.launcher = BasicLauncher(sys_info=self.configs.get('sys_info', {"name": "local"}), launcher_config=self.configs.get('launcher', {"mode": "mpi"}))

    def register_component(self, name: str, 
                          executable: Union[str, Callable], 
                          type: str,
                          args: Dict[str, Any] = None,
                          nodes: List[str] = None,
                          ppn: int = 1,
                          num_gpus_per_process: int = 0,
                          cpu_affinity: List[int] = None,
                          gpu_affinity: List[str] = None, 
                          env_vars: Dict[str, str] = None,
                          dependencies: List[str] = None) -> 'Workflow':
        """
        Register a component in the workflow.
        
        Args:
            name: Unique name for this component
            executable: Command string or Python function to execute
            type: Component type ("local", "remote", "dragon")
            args: Arguments dictionary for the component
            nodes: List of nodes to run on (optional)
            ppn: Processes per node (optional)
            num_gpus_per_process: Number of GPUs per process (optional)
            cpu_affinity: CPU cores to bind to (optional)
            gpu_affinity: GPU devices to bind to (optional)
            env_vars: Environment variables (optional)
            dependencies: List of component names this depends on (optional)
            
        Returns:
            Self for method chaining
        """
        component = WorkflowComponent(
            name=name,
            type=type,
            executable=executable,
            args=args or {},
            nodes=nodes or [],
            ppn=ppn,
            num_gpus_per_process=num_gpus_per_process,
            cpu_affinity=cpu_affinity,
            gpu_affinity=gpu_affinity,
            env_vars=env_vars or {},
            dependencies=dependencies or []
        )
        
        self.components[name] = component
        return self

    def get_component(self, name: str) -> Optional[WorkflowComponent]:
        """Get a registered component by name."""
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self.components.keys())

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
                 type: str = "local",
                 args: Dict[str, Any] = None,
                 nodes: List[str] = None,
                 ppn: int = 1,
                 num_gpus_per_process: int = 0,
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
            type: Component type ("local", "remote", "dragon")
            args: Arguments dictionary for the component
            nodes: List of nodes to run on (optional)
            ppn: Processes per node (optional)
            num_gpus_per_process: Number of GPUs per process (optional)
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
                
            @workflow.component(name="my_task", args={"--input": "file.txt"}, dependencies=["setup"])
            def third_function():
                return 0
        """
        def decorator(f: Callable):
            component_name = name if name is not None else f.__name__
            self.register_component(
                name=component_name,
                type=type,
                executable=f,
                args=args,
                nodes=nodes,
                ppn=ppn,
                num_gpus_per_process=num_gpus_per_process,
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