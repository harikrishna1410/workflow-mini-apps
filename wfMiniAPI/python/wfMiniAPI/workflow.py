from training import AI
from simulation import Simulation
import numpy as np

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

# ##Autiomatically set the number of layers based on the training time in the workflow
# ai = AI()
# train_time = 1
# ai.set_nlayers_train(train_time)
# print(f"number of layers {ai.model.nlayers}")


sim = Simulation()
### Add kernels to the simulation
sim.add_kernel("kernel1", matmul , run_count=1, data_size=(8,8,8))
sim.add_kernel("kernel2", x2, run_count=1, data_size=(8,8,8))
sim.add_kernel("kernel3", x3, run_count=1, data_size=(8,8,8))

##set the run count of each kernel to 0.5 seconds
sim.set_kernel_run_count_by_time("kernel1", 0.5)
sim.set_kernel_run_count_by_time("kernel2", 0.5)
sim.set_kernel_run_count_by_time("kernel3", 0.5)

for k in sim.kernels:
    print(f"Kernel: {k['name']}, Run Count: {k['run_count']}, Data Size: {k['data_size']}")