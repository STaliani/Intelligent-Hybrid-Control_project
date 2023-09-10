import numpy as np
import matplotlib.pyplot as plt
from dmd_perturbation_detection.dataset_builder import create_dataset
from dmd_perturbation_detection.robot import Robot
from dmd_perturbation_detection import dmd_class, trajectories, gpr

# Decide the number of data points
N_samples = 5

# initialize initial and final position for the robot
start_points = np.array([[0.1, 0.2], [1., 0.], [ 0.5, 0.5], [-1.1, 0.2], [0.8, -0.2]]).transpose()
end_points   = np.array([[0.2, 0. ], [0., 1.], [-0.5, 1.3], [-0.3,-1.1], [0.1, -0.7]]).transpose()
execution_time = np.array([10., 10., 10., 10., 10.])

# initialize a 2R planar robot
rbt = Robot(1.,1.)
dmd = dmd_class.DMD()
# create the dataset, ground truth + noise, first row is the ground truth 
dataset, disturbance_ground_truth = create_dataset(rbt, start_points, end_points, execution_time, N_samples)

for key in dataset.keys():
    plt.figure("Torques "+  key)
    #plot the ground truth
    plt.plot(dataset[key][0,:], 'b') 
    plt.plot(dataset[key][1,:], 'orange')
     
    # plot the noisy data
    #plt.plot(dataset[key][2,:], 'ob')
    #plt.plot(dataset[key][3,:], 'o', color='orange')
    
    #plt.plot(dataset[key][4,:], 'ob')
    #plt.plot(dataset[key][5,:], 'o', color='orange')
    
    #plt.plot(dataset[key][6,:], 'ob')
    #plt.plot(dataset[key][7,:], 'o', color='orange')
    
    plt.plot(dataset[key][8,:], 'ob')
    plt.plot(dataset[key][9,:], 'o', color='orange')
    
    plt.xlabel("Time")
    plt.ylabel("Torque")
    plt.legend(["Joint 1", "Joint 2"])
    plt.grid()
    scores = []
    A_tilde, Phi, A = dmd.dmd(dataset[key][:2,:], 2)
    for sample in range(0,N_samples+1):
        N = dataset[key].shape[1]
        prediction = np.zeros((2,N))
        prediction[:,0] = dataset[key][:2,0]
        score = 0
        for i in range(1,N):
            tau_next = dataset[key][:2,i:i+1]
            tau_predicted = dmd.dmd4cast(dataset[key][2*sample:2*sample+2,:i], 2, 1)
            prediction[:,i:i+1] = tau_predicted
            a = np.greater((tau_next - tau_predicted), np.ones((2,1))*0.2)
            b = np.less((tau_next - tau_predicted), -np.ones((2,1))*0.2)
            if a.all() or b.all():
                print(f"anomaly detected at sample {sample} and time {i}")
                if disturbance_ground_truth[key][sample,i] == 1:
                    score += 1
            else:
                if disturbance_ground_truth[key][sample,i] == 0:
                    score += 1
        print(f"Score for sample {sample} is {score/N}")
        scores.append(score/N)
        plt.figure(f"Predicted torques error {key} sample {sample}")
        plt.plot(prediction[0,:]- dataset[key][0,:], 'b')
        plt.plot(prediction[1,:]- dataset[key][1,:], 'orange')
        plt.xlabel("Time")
        plt.ylabel("Torque")
        plt.legend(["Joint 1", "Joint 2"])
        plt.grid()
        #plt.show()
    plt.figure("Scores")
    plt.step(np.arange(0,N_samples+1), scores)
    plt.xlabel("Sample")
    plt.ylabel("Score")
    plt.grid()
    plt.show()
    
    #pred_step = 2
    #next_tau = dmd.DMD4cast(dataset[key], 1, pred_step)
    
    
    
plt.show()