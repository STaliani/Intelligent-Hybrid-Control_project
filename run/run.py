import numpy as np
import matplotlib.pyplot as plt
from dmd_perturbation_detection.dataset_builder import create_dataset
from dmd_perturbation_detection.robot import Robot
from dmd_perturbation_detection import dmd_class, trajectories, gpr
from dmd_perturbation_detection.animation import animate

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
dataset, disturbance_ground_truth, ext_force, traj = create_dataset(rbt, start_points, end_points, execution_time, N_samples)
detected_dist = dict.fromkeys(["traj_" + str(key) for key in range(N_samples)])
for key in dataset.keys():
    plt.figure("Torques "+  key)
    #plot the ground truth
    plt.plot(dataset[key][0,:], 'b') 
    plt.plot(dataset[key][1,:], 'orange')
     
    # plot the noisy 
    #plt.figure("Torques "+  key + " sample 1")
    plt.plot(dataset[key][2,:], 'ob')
    plt.plot(dataset[key][3,:], 'o', color='orange')
    
    #plt.figure("Torques "+  key + " sample 2")
    plt.plot(dataset[key][4,:], 'ob')
    plt.plot(dataset[key][5,:], 'o', color='orange')
    
    #plt.figure("Torques "+  key + " sample 3")
    plt.plot(dataset[key][6,:], 'ob')
    plt.plot(dataset[key][7,:], 'o', color='orange')
    
    #plt.figure("Torques "+  key + " sample 4")
    plt.plot(dataset[key][8,:], 'ob')
    plt.plot(dataset[key][9,:], 'o', color='orange')
    
    #plt.figure("Torques "+  key + " sample 5")
    plt.plot(dataset[key][10,:], 'ob')
    plt.plot(dataset[key][11,:], 'o', color='orange')
    
    plt.xlabel("Time")
    plt.ylabel("Torque")
    plt.legend(["Joint 1", "Joint 2"])
    plt.grid()
    scores = []
    confusion = dict.fromkeys(['TP', 'FP', 'TN', 'FN'])
    for entry in confusion.keys():
        confusion[entry] = 0
    A_tilde, Phi, A = dmd.dmd(dataset[key][:2,:], 2)
    detection = np.zeros((N_samples,dataset[key].shape[1]))
    for sample in range(1,N_samples+1):
        N = dataset[key].shape[1]
        prediction = np.zeros((2,N))
        prediction[:,0] = dataset[key][:2,0]
        score = 0
        for i in range(1,N):
            tau_next = dataset[key][2*sample:2*sample+2,i:i+1]
            tau_predicted = dmd.dmd4cast(dataset[key][0:2,:i], 2, 1)
            prediction[:,i:i+1] = tau_predicted
            error = tau_next - tau_predicted
            a = error >= np.ones((2,1))*0.2
            #print('a',a)
            b = error <= -np.ones((2,1))*0.2
            #print('b',b)
            if (a[0] or a[1]) or (b[0] or b[1]):
                detection[sample-1,i] = 1
                #print(f"anomaly detected at sample {sample} and time {i}")
                if disturbance_ground_truth[key][sample,i] == 1:
                    score += 1
                    confusion['TP'] += 1
                else:
                    confusion['FP'] += 1
            else:
                if disturbance_ground_truth[key][sample,i] == 0:
                    score += 1
                    confusion['TN'] += 1
                else:
                    confusion['FN'] += 1

        scores.append(score/(N-1))
        animation = animate(rbt, traj[key], detection[sample-1,:], ext_force[key][sample-1,:])

    #animation.save(f"animations/animation_{key}.gif", writer='imagemagick', fps=10)
    detected_dist[key] = detection      
    conf_matrix = np.array([[confusion['TP'], confusion['FP']], [confusion['FN'], confusion['TN']]])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    #fig.savefig(f"animations/Confusion Matrix {key}")
    
    scores = np.array(scores)
    plt.figure("Prediction")
    plt.bar(list(range(5)),scores)
    plt.xlabel("sample")
    plt.ylabel("Probability")    
    plt.grid()
    #plt.savefig(f"animations/Prediction {key}")
    plt.show()