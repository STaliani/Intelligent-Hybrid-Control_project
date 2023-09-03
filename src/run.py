import numpy as np
from robot import Robot
import trajectories
import matplotlib.pyplot as plt
import dmd_class
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
dmd = dmd_class.DMD()
# create the dataset, ground truth + noise, first row is the ground truth 
dataset = create_dataset(rbt, start_points, end_points, execution_time, N_samples)

for key in dataset.keys():
    plt.figure("Torques "+  key)
    #plot the ground truth
    plt.plot(dataset[key][0,:], 'b') 
    plt.plot(dataset[key][1,:], 'orange')
     
    # plot the noisy data
    plt.plot(dataset[key][2,:], 'ob')
    plt.plot(dataset[key][3,:], 'o', color='orange')
    
    plt.plot(dataset[key][4,:], 'ob')
    plt.plot(dataset[key][5,:], 'o', color='orange')
    
    plt.plot(dataset[key][6,:], 'ob')
    plt.plot(dataset[key][7,:], 'o', color='orange')
    
    plt.plot(dataset[key][8,:], 'ob')
    plt.plot(dataset[key][9,:], 'o', color='orange')
    
    plt.xlabel("Time")
    plt.ylabel("Torque")
    plt.legend(["Joint 1", "Joint 2"])
    plt.grid()
    
    A_tilde, Phi, A = dmd.dmd(dataset[key][:2,:], 2)
    for sample in range(1,N_samples+1):
        N = dataset[key].shape[1]
        for i in range(1,N):
            tau_next = dataset[key][:2,i:i+1]
            tau_predicted = dmd.dmd4cast(dataset[key][2*sample:2*sample+2,:i], 2, 1)
            a = np.greater((tau_next - tau_predicted), np.ones((2,1))*0.2)
            b = np.less((tau_next - tau_predicted), -np.ones((2,1))*0.2)
            if a.all() or b.all():
                print(f"anomaly detected at sample {sample} and time {i}")
    
    #pred_step = 2
    #next_tau = dmd.DMD4cast(dataset[key], 1, pred_step)
    
    
    
plt.show()