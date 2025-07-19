from wfMiniAPI import Workflow,Simulation
import os, json
import logging


def test_workflow():
    my_workflow = Workflow(launcher={"mode": "high throughput"})
    @my_workflow.component(name="sim")
    def run_simulation():
        sim = Simulation(logging=True, log_level=logging.DEBUG)
        ##add two kernels to the simulation
        sim.add_kernel("MatMulSimple2D", run_time=1.0)
        sim.add_kernel("MatMulSimple2D", run_time=1.0)
        sim.run()
    
    @my_workflow.component(name="sim2")
    def sim2():
        sim = Simulation(logging=True, log_level=logging.DEBUG)
        sim.add_kernel("MatMulGeneral", run_count=10)
        sim.run()

    assert my_workflow.launch() == 0


if __name__ == "__main__":
    test_workflow()
    print("Workflow test passed!")
