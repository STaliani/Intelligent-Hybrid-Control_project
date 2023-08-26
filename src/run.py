import numpy as np
from robot import Robot
import trajectories
import matplotlib.pyplot as plt
from dmd import DMD, DMD4cast
from create_dataset import dataset_builder

# Decide the number of data points
N_samples = 5

# initialize initial and final position for the robot
start_points = np.array([[0.1, 0.2], [1., 0.], [ 0.5, 0.5], [-1.1, 0.2], [0.8, -0.2]]).transpose()
end_points   = np.array([[0.2, 0. ], [0., 1.], [-0.5, 1.3], [-0.3,-1.1], [0.1, -0.7]]).transpose()
execution_time = np.array([5., 5., 5., 5., 5.])

# initialize a 2R planar robot
rbt = Robot(1.,1.)

# create the dataset
dataset = dataset_builder(rbt, start_points, end_points, execution_time, N_samples)

print(dataset.keys())

for key in dataset.keys():
    plt.figure("Torques "+  key)
    plt.plot(dataset[key][0,:], 'o')
    plt.plot(dataset[key][1,:], 'o')
    plt.xlabel("Time")
    plt.ylabel("Torque")
    plt.legend(["Joint 1", "Joint 2"])
    plt.grid()
    
    #pred_step = 2
    #next_tau = DMD4cast(dataset[key], 2, pred_step)
    #print(next_tau.shape)
    A_tilde, Phi, A = DMD(dataset[key], 2)
    #print("A_tilde:\n", A_tilde, "\nPhi:\n", Phi, "\nA:\n",A)
    print("A_tilde: ", A_tilde.shape, "\nPhi: ", Phi.shape, "\nA: ",A.shape)
    
#plt.show()