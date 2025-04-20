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
#create 10 AI models

AIs = []
for i in range(10):
    ##build the AI model
    ai = AI(name=f"ai_{i}",num_layers=10)
    AIs.append(ai)

##create 10 simulations
simulations = []
for i in range(10):
    sim = Simulation(name=f"sim_{i}")
    ### Add kernels to the simulation
    sim.add_kernel("kernel1", matmul , run_count=100, data_size=(16,16,16))
    simulations.append(sim)

##************done building the workflow***********************

##*************run the workflow********************************
##main workflow
niters = 1
for i in range(niters):
    ##run all the simulations
    for sim_id,sim in enumerate(simulations):
        ##run the simulation
        sim.run()
        ##data from each simulation model will be used to 
        # train ai model index <= sim_id models 
        ##train the AI model
        sim.stage_write(f"sim_{sim_id}",f",".join([f"train_{i}" for i in range(sim_id+1)]))
    ##read data from all simulations
    for sim_id,sim in enumerate(simulations):
        for ai_id,ai in enumerate(AIs):
            data = ai.stage_read(f"sim_{sim_id}")
            if f"train_{ai_id}" in data:
                print(f"Training AI model {ai.name} using data from {sim.name}")
                ai.train()

##*************done running the workflow***********************