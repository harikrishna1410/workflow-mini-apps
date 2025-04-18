from training import AI
from simulation import Simulation
import numpy as np


#**********kernel functions***************
def matmul(data_size):
    a = np.random.rand(*data_size)
    b = np.random.rand(*data_size)
    # Perform matrix multiplication
    c = np.dot(a, b)
    return None

def x2(data_size):
    a = np.random.rand(*data_size)
    # Perform element-wise square
    c = np.square(a)
    return None 

def x3(data_size):
    a = np.random.rand(*data_size)
    # Perform element-wise cube
    c = np.power(a, 3)
    return None

###*************build the workflow*******************************************
##build the AI model
ai = AI()
train_time = 0.5
ai.set_nlayers_train(train_time)
print(f"number of layers in the AI models {ai.model.nlayers}")


#Build the simulation
sim = Simulation()
### Add kernels to the simulation
sim.add_kernel("kernel1", matmul , run_count=1, data_size=(8,8,8))
sim.add_kernel("kernel2", x2, run_count=1, data_size=(16,16,16))
sim.add_kernel("kernel3", x3, run_count=1, data_size=(16,16,16))

##set the run count of each kernel to 0.5 seconds
sim.set_kernel_run_count_by_time("kernel1", 1)
sim.set_kernel_run_count_by_time("kernel2", 1)
sim.set_kernel_run_count_by_time("kernel3", 1)


for k in sim.kernels:
    print(f"Kernel: {k['name']}, Run Count: {k['run_count']}, Data Size: {k['data_size']}")

#connect AI and simulation
sim.connect(ai)
ai.connect(sim)
##************done building the workflow***********************

##*************run the workflow********************************
##main workflow
niters = 10
for i in range(niters):
    ##run the simulation
    sim.run()
    ##send data to AI
    sim.send("sim data")
    ##received data from SIM
    ai.receive()
    ##train the AI model
    ai.train()
##*************done running the workflow***********************