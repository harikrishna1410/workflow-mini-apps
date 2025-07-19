import os
import multiprocessing
import socket
from typing import List, Union, Dict, Any, Callable, Tuple, Optional

# Handle Dragon imports and type hints
try:
    import dragon
    from dragon.native.process_group import ProcessGroup
    from dragon.native.process import ProcessTemplate, Process as DragonProcess, Popen
    from dragon.infrastructure.policy import Policy
    DRAGON_AVAILABLE = True
except ImportError:
    DRAGON_AVAILABLE = False
    # When Dragon is not available, we'll use Any for type hints
    ProcessGroup = Any

# Try to import ensemble_launcher components
try:
    from ensemble_launcher.helper_functions import create_task_info
    from ensemble_launcher.worker import worker
    ENSEMBLE_LAUNCHER_AVAILABLE = True
except ImportError:
    raise ImportError("ensemble_launcher is required but not available. Please install ensemble_launcher to use this launcher.")


class BasicLauncher:
    """
    Ensemble launcher-based workflow component launcher.
    
    This launcher uses ensemble_launcher as the primary execution engine for all
    workflow components. It provides unified execution for local and remote tasks
    with advanced resource management and system-specific optimizations.
    
    Requires ensemble_launcher to be installed.
    """
    
    def __init__(self, system: str = "local", launcher_config: Dict[str, Any] = None):
        """
        Initialize the BasicLauncher.
        
        Args:
            system: System name (e.g., "local", "aurora", "polaris")
            launcher_config: Configuration for the launcher
        """
        self.system = system
        self.launcher_config = launcher_config or {"mode": "mpi"}
        
    @staticmethod
    def _prepare_environment(base_env: Dict[str, str] = None, 
                           additional_env: Dict[str, str] = None) -> Dict[str, str]:
        """
        Prepare environment variables for process execution.
        
        Args:
            base_env: Base environment (defaults to os.environ)
            additional_env: Additional environment variables
        
        Returns:
            Environment dictionary
        """
        env = (base_env or os.environ).copy()
        
        if additional_env:
            env.update(additional_env)
            
        return env
    
    def _workflow_component_to_task_info(self, workflow_component, task_id: str = None) -> Dict[str, Any]:
        """
        Convert a workflow component to ensemble_launcher task_info format.
        
        Args:
            workflow_component: WorkflowComponent object
            task_id: Optional task ID (defaults to component name)
            
        Returns:
            Dictionary in task_info format
        """
        if task_id is None:
            task_id = getattr(workflow_component, 'name', f"task_{id(workflow_component)}")
        
        # Handle both string executables and Python callables
        if isinstance(workflow_component.executable, str):
            cmd_template = workflow_component.executable
        else:
            # For Python callables, we need to create a shell command that can execute them
            # since ensemble_launcher uses cmd_template for actual task execution
            import base64
            import pickle
            
            # Try to use cloudpickle for better serialization, fallback to pickle
            try:
                import cloudpickle
                serialized_func = cloudpickle.dumps(workflow_component.executable)
                pickle_module = "cloudpickle"
            except ImportError:
                serialized_func = pickle.dumps(workflow_component.executable)
                pickle_module = "pickle"
            
            try:
                encoded_func = base64.b64encode(serialized_func).decode('ascii')
                
                # Create command that deserializes and executes the function
                cmd_template = (
                    f"python3 -c \""
                    f"import base64; import {pickle_module}; "
                    f"func = {pickle_module}.loads(base64.b64decode('{encoded_func}')); "
                    f"func()\""
                )
            except Exception:
                # Final fallback - assume it's a simple function call
                func_name = getattr(workflow_component.executable, '__name__', 'unknown_function')
                module_name = getattr(workflow_component.executable, '__module__', '__main__')
                if module_name == '__main__':
                    # If function is from __main__, we can't import it this way
                    cmd_template = f"python3 -c \"print('Error: Cannot execute function {func_name} from __main__ module')\""
                else:
                    cmd_template = f"python3 -c \"from {module_name} import {func_name}; {func_name}()\""
        
        # Calculate nodes list
        nodes = workflow_component.nodes if hasattr(workflow_component, 'nodes') and workflow_component.nodes else [socket.gethostname()]
        num_nodes = len(nodes)
        
        # Get processes per node
        ppn = getattr(workflow_component, 'ppn', 1)
        
        # Get GPU information
        num_gpus_per_process = 0
        gpu_affinity = None
        if hasattr(workflow_component, 'gpu_affinity') and workflow_component.gpu_affinity:
            gpu_affinity = workflow_component.gpu_affinity
            num_gpus_per_process = len(gpu_affinity) // (num_nodes * ppn) if len(gpu_affinity) > 0 else 0
        
        # Get CPU affinity
        cpu_affinity = getattr(workflow_component, 'cpu_affinity', None)
        
        # Get environment variables
        env_vars = getattr(workflow_component, 'env_vars', {})
        
        # Create task_info using the helper function
        task_info = create_task_info(
            task_id=task_id,
            cmd_template=cmd_template,
            system=self.system,
            num_nodes=num_nodes,
            num_processes_per_node=ppn,
            num_gpus_per_process=num_gpus_per_process,
            gpu_affinity=gpu_affinity,
            cpu_affinity=cpu_affinity,
            env=env_vars,
            run_dir=getattr(workflow_component, 'run_dir', None),
            launch_dir=getattr(workflow_component, 'launch_dir', None),
            timeout=getattr(workflow_component, 'timeout', None)
        )
        
        # Add assigned nodes for the worker
        task_info['assigned_nodes'] = nodes
        task_info['assigned_cores'] = {node: list(range(ppn)) for node in nodes}
        task_info['assigned_gpus'] = {node: gpu_affinity[:num_gpus_per_process] if gpu_affinity else [] for node in nodes}
        
        return task_info
    
    def _launch_component_with_ensemble(self, workflow_component) -> multiprocessing.Process:
        """
        Launch a single component using ensemble_launcher worker in a multiprocessing.Process.
        Handles both local and remote execution types through ensemble launcher.
        
        Args:
            workflow_component: WorkflowComponent object to launch
            
        Returns:
            multiprocessing.Process object representing the launched worker process
        """
        # Convert workflow component to task_info
        task_id = getattr(workflow_component, 'name', f"task_{id(workflow_component)}")
        task_info = self._workflow_component_to_task_info(workflow_component, task_id)
        
        # Create tasks dictionary
        my_tasks = {task_id: task_info}

        # Get nodes list
        nodes = workflow_component.nodes if hasattr(workflow_component, 'nodes') and workflow_component.nodes else [socket.gethostname()]
        
        # Create system info
        sys_info = {
            "name": self.system,
            "ncores_per_node": getattr(workflow_component, 'ncores_per_node', multiprocessing.cpu_count()),
            "ngpus_per_node": getattr(workflow_component, 'ngpus_per_node', 0)
        }
        
        # Create worker instance
        worker_id = f"worker_{task_id}"
        comm_config = {"comm_layer": "multiprocessing"}
        
        worker_instance = worker(
            worker_id=worker_id,
            my_tasks=my_tasks,
            my_nodes=nodes,
            sys_info=sys_info,
            comm_config=comm_config,
            launcher_config=self.launcher_config
        )
        
        # Create and start the multiprocessing.Process directly with worker.run_tasks
        process = multiprocessing.Process(
            target=worker_instance.run_tasks,
            kwargs={'logger': True},
            name=f"worker_{task_id}"
        )
        
        # Store reference to task info for later retrieval
        process.task_id = task_id
        process.worker_instance = worker_instance
        
        # Start the process
        process.start()
        
        return process
    
    @staticmethod
    def _launch_dragon_component(workflow_component, dragon_dict=None) -> Any:
        """
        Launch a single component using Dragon.
        
        Args:
            workflow_component: WorkflowComponent object to launch
            dragon_dict: Dragon dictionary for inter-process communication
            
        Returns:
            Launched process group
        """
        if not DRAGON_AVAILABLE:
            raise RuntimeError("Dragon is not available")
        
        if not callable(workflow_component.executable):
            raise ValueError("Dragon launcher requires Python callable, not string executable")
        
        # Calculate total processes
        total_processes = workflow_component.ppn
        if workflow_component.nodes:
            total_processes = workflow_component.ppn * len(workflow_component.nodes)
        
        # Create process group
        policy = Policy(distribution=Policy.Distribution.BLOCK)
        pg = ProcessGroup(pmi_enabled=False, policy=policy)
        
        # Add processes based on nodes and ppn
        process_count = 0
        for nid, node in enumerate(workflow_component.nodes or ["localhost"]):
            for local_rank in range(workflow_component.ppn):
                if process_count >= total_processes:
                    break
                
                # Create placement policy
                placement_policy = Policy(
                    placement=Policy.Placement.HOST_NAME,
                    host_name=node
                )
                
                # Set CPU affinity
                if workflow_component.cpu_affinity and local_rank < len(workflow_component.cpu_affinity):
                    placement_policy.cpu_affinity = [workflow_component.cpu_affinity[local_rank]]
                
                # Set GPU affinity
                if workflow_component.gpu_affinity and local_rank < len(workflow_component.gpu_affinity):
                    placement_policy.gpu_affinity = [workflow_component.gpu_affinity[local_rank]]
                
                # Prepare environment
                env = BasicLauncher._prepare_environment(
                    additional_env=workflow_component.env_vars
                )
                
                # Create process template
                template_args = []
                if dragon_dict is not None:
                    template_args.append(dragon_dict)
                
                pg.add_process(
                    nproc=1,
                    template=ProcessTemplate(
                        target=workflow_component.executable,
                        args=tuple(template_args),
                        cwd=os.path.dirname(__file__),
                        policy=placement_policy,
                        stdout=Popen.DEVNULL,
                        env=env
                    )
                )
                
                process_count += 1
            
            if process_count >= total_processes:
                break
        
        pg.init()
        pg.start()
        
        return pg
    

    @staticmethod
    def _wait_for_dragon_processes(process_groups: List[Any], timeout: int = 30):
        """
        Wait for Dragon process groups to complete.
        
        Args:
            process_groups: List of process groups to wait for
            timeout: Timeout for joining process groups
        """
        for pg in process_groups:
            try:
                pg.join(timeout)
            except Exception as e:
                pg.stop()
            finally:
                pg.close()

    def launch_component(self, workflow_component) -> Union[multiprocessing.Process, Any]:
        """
        Launch a single workflow component using ensemble_launcher or Dragon.
        
        Args:
            workflow_component: WorkflowComponent object to launch
            
        Returns:
            multiprocessing.Process object for ensemble launcher or ProcessGroup for Dragon
        """
        component_type = getattr(workflow_component, 'type', 'ensemble')
        
        if component_type == "dragon":
            return self._launch_dragon_component(workflow_component)
        else:
            # Use ensemble launcher for all other types (local, remote, ensemble)
            return self._launch_component_with_ensemble(workflow_component)
    
    def wait_for_component(self, launched_process, timeout: int = None) -> Union[int, List[int]]:
        """
        Wait for a launched workflow component to complete.
        
        Args:
            launched_process: The process or process group to wait for
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Exit code or list of exit codes
        """
        if isinstance(launched_process, multiprocessing.Process):
            # This is a multiprocessing.Process from ensemble launcher
            try:
                launched_process.join(timeout=timeout)
                return launched_process.exitcode if launched_process.exitcode is not None else 0
            except Exception as e:
                # If join times out or fails, terminate the process
                if launched_process.is_alive():
                    launched_process.terminate()
                    launched_process.join(1)  # Give it 1 second to terminate gracefully
                    if launched_process.is_alive():
                        launched_process.kill()  # Force kill if still alive
                return launched_process.exitcode if launched_process.exitcode is not None else 1
        elif DRAGON_AVAILABLE and ProcessGroup is not None and isinstance(launched_process, ProcessGroup):
            self._wait_for_dragon_processes([launched_process], timeout or 30)
            return [0]  # Dragon doesn't provide exit codes in the same way
        else:
            raise ValueError(f"Unknown launched process type: {type(launched_process)}. Expected multiprocessing.Process, task_info dict, or Dragon ProcessGroup.")

    def get_task_result(self, launched_process: multiprocessing.Process) -> Dict[str, Any]:
        """
        Get the task result from a completed ensemble launcher multiprocessing.Process.
        
        Args:
            launched_process: The multiprocessing.Process object returned by launch_component
            
        Returns:
            Task info dictionary with execution results
        """
        if not isinstance(launched_process, multiprocessing.Process):
            raise ValueError("get_task_result only supports multiprocessing.Process objects from ensemble launcher")
        
        if launched_process.is_alive():
            raise RuntimeError("Process is still running. Call wait_for_component() first.")
        
        # Return the task info from the worker instance
        # Since the process has completed, we can access the final state
        task_id = getattr(launched_process, 'task_id', None)
        worker_instance = getattr(launched_process, 'worker_instance', None)
        
        if task_id and worker_instance and hasattr(worker_instance, 'my_tasks'):
            return worker_instance.my_tasks.get(task_id, {})
        else:
            # Fallback - create a basic result based on exit code
            return {
                'status': 'finished' if launched_process.exitcode == 0 else 'failed',
                'returncode': launched_process.exitcode,
                'execution_time': 0.0
            }

    @classmethod
    def create_ensemble_launcher(cls, system: str = "local", launcher_mode: str = "mpi", **kwargs):
        """
        Create a BasicLauncher configured to use ensemble_launcher by default.
        
        Args:
            system: System name (e.g., "local", "aurora", "polaris")
            launcher_mode: Launcher mode ("mpi" or "high throughput")
            **kwargs: Additional launcher configuration options
            
        Returns:
            BasicLauncher instance configured for ensemble launcher
        """
        launcher_config = {"mode": launcher_mode}
        launcher_config.update(kwargs)
        
        return cls(system=system, launcher_config=launcher_config)
    
    def get_task_status(self, task_result: Union[Dict[str, Any], multiprocessing.Process]) -> str:
        """
        Get the status of a task from ensemble launcher result.
        
        Args:
            task_result: Task info dictionary or multiprocessing.Process object
            
        Returns:
            Task status string ("finished", "failed", "running", etc.)
        """
        if isinstance(task_result, multiprocessing.Process):
            if task_result.is_alive():
                return "running"
            else:
                return "finished" if task_result.exitcode == 0 else "failed"
        elif isinstance(task_result, dict) and 'status' in task_result:
            return task_result['status']
        return "unknown"
    
    def get_task_execution_time(self, task_result: Union[Dict[str, Any], multiprocessing.Process]) -> float:
        """
        Get the execution time of a task from ensemble launcher result.
        
        Args:
            task_result: Task info dictionary or multiprocessing.Process object
            
        Returns:
            Execution time in seconds, or 0.0 if not available
        """
        if isinstance(task_result, multiprocessing.Process):
            # For multiprocessing.Process, we need to get the actual task result
            if not task_result.is_alive():  # Process has completed
                actual_result = self.get_task_result(task_result)
                return actual_result.get('execution_time', 0.0)
            return 0.0  # Still running
        elif isinstance(task_result, dict) and 'execution_time' in task_result:
            return task_result['execution_time']
        return 0.0


# Example usage:
"""
To use the ensemble launcher integration (ensemble_launcher is required):

from wfMiniAPI.launcher import BasicLauncher

# Create a launcher configured for ensemble execution
launcher = BasicLauncher.create_ensemble_launcher(system="local", launcher_mode="mpi")

# Or create with specific configuration
launcher = BasicLauncher(
    system="aurora", 
    launcher_config={"mode": "high throughput", "cpu_bind": True}
)

# Launch a workflow component
# The launcher now uses ensemble_launcher for all component types (local, remote, ensemble)
# Assuming you have a workflow_component object with the necessary attributes:
# - executable: str or callable
# - name: str
# - ppn: int (processes per node)
# - nodes: List[str] (optional)
# - gpu_affinity: List[str] (optional)
# - cpu_affinity: List[int] (optional)
# - env_vars: Dict[str, str] (optional)
# - type: str ("local", "remote", "dragon", or "ensemble") - defaults to "ensemble"

# Launch returns a multiprocessing.Process object (non-blocking)
process = launcher.launch_component(workflow_component)

# Check status while running
status = launcher.get_task_status(process)
print(f"Task status: {status}")

# Wait for completion (returns exit code)
exit_code = launcher.wait_for_component(process)

# Get detailed task results after completion
task_result = launcher.get_task_result(process)
exec_time = launcher.get_task_execution_time(task_result)

print(f"Task completed with exit code: {exit_code}")
print(f"Execution time: {exec_time}s")

Note: This launcher requires ensemble_launcher to be installed. 
Dragon components are still supported if dragon is available.
All local and remote components are now executed through ensemble_launcher as 
non-blocking multiprocessing.Process objects for better resource management and system optimization.
"""
