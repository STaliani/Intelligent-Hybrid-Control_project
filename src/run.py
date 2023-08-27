import numpy as np
from robot import Robot
import trajectories
import matplotlib.pyplot as plt
import dmd
from dataset_builder import create_dataset
import gpr

# Decide the number of data points
N_samples = 5

# initialize initial and final position for the robot
start_points = np.array([[0.1, 0.2], [1., 0.], [ 0.5, 0.5], [-1.1, 0.2], [0.8, -0.2]]).transpose()
end_points   = np.array([[0.2, 0. ], [0., 1.], [-0.5, 1.3], [-0.3,-1.1], [0.1, -0.7]]).transpose()
execution_time = np.array([5., 5., 5., 5., 5.])

# initialize a 2R planar robot
rbt = Robot(1.,1.)

# create the dataset, ground truth + noise, first row is the ground truth 
dataset = create_dataset(rbt, start_points, end_points, execution_time, N_samples)
# Initialize the Gaussian Process Regressor
regressor = gpr.GPR(kernel= None)
# Train the regressor
regressor.train_regressor(dataset["traj_0"])

for key in dataset.keys():
    plt.figure("Torques "+  key)
    #plot the ground truth
    plt.plot(dataset[key][0,:], 'b') 
    plt.plot(dataset[key][1,:], 'orange') 
    # plot the noisy data
    plt.plot(dataset[key][2,:], 'ob') 
    plt.plot(dataset[key][3,:], 'o', color='orange')
    plt.xlabel("Time")
    plt.ylabel("Torque")
    plt.legend(["Joint 1", "Joint 2"])
    plt.grid()
    
    data = regressor.extract_evolution_from_measure(dataset[key][2:4,:])
    
    pred_step = 2
    next_tau = dmd.DMD4cast(dataset[key], 1, pred_step)
    
    
    
#plt.show()