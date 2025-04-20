from wfMiniAPI import AI, Simulation
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
ai = AI(num_layers=100)

#Build the simulation
sim = Simulation()
### Add kernels to the simulation
sim.add_kernel("kernel1", matmul , run_count=100, data_size=(32,32,32))

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
    if i % 2 == 0:
        ##train the AI model
        sim.send(f"train")
    else:
        sim.send(f"infer")
    ##received data from SIM
    data = ai.receive()
    if data["SIM"] == "train":
        ##train the AI model
        print(f"Training AI model for {ai.num_epochs} epochs")
        ai.train()
    else:
        print("Infer from the AI model")
        ai.infer()
##*************done running the workflow***********************