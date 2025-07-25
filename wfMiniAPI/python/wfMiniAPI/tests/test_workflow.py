from wfMiniAPI import Workflow,Simulation
import os, json
import logging

class RC:
    def __init__(self, data:int):
        self.data = data

def test_workflow():
    my_workflow = Workflow(launcher={"mode":"high throughput"})
    @my_workflow.component(name="sim",type="remote",args={"runcount": RC(10)})
    def run_simulation(runcount:RC=RC(10)):
        sim = Simulation(name="sim", logging=True, log_level=logging.DEBUG)
        ##add two kernels to the simulation
        sim.add_kernel("MatMulSimple2D", run_count=runcount.data)
        sim.run()

    @my_workflow.component(name="sim2", type="local",args={"runcount": 22})
    def sim2(runcount=10):
        sim = Simulation(name="sim2", logging=True, log_level=logging.DEBUG)
        sim.add_kernel("MatMulGeneral", run_count=runcount)
        sim.run()

    # my_workflow.register_component(name="sim3",executable="echo 'Hello from sim3'", type="local",dependencies=["sim","sim2"])
    assert my_workflow.launch() == 0


if __name__ == "__main__":
    test_workflow()
    print("Workflow test passed!")
