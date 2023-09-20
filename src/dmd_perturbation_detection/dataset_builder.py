import dmd_perturbation_detection.trajectories as trajectories
from dmd_perturbation_detection.robot import Robot
import numpy as np
import matplotlib.pyplot as plt
import copy

def create_dataset(rbt: Robot, start_points:np.ndarray, end_points:np.ndarray, execution_time:np.ndarray, N_samples:int, noise_level:float = 0.1)->dict:
    _, n = start_points.shape
    dataset = dict.fromkeys(["traj_" + str(key) for key in range(N_samples)])
    disturbance_ground_truth = dict.fromkeys(["traj_" + str(key) for key in range(N_samples)])
    force_ground_truth = dict.fromkeys(["traj_" + str(key) for key in range(N_samples)])
    for key in force_ground_truth.keys():
        force_ground_truth[key] = np.zeros((5,3))
    for point in range(n):
        # create the trajectory
        start_q = rbt.inverseKinimatics(start_points[0,point], start_points[1,point])
        end_q = rbt.inverseKinimatics(end_points[0,point], end_points[1,point])
        traj = trajectories.createTrajectory(start_q, end_q, execution_time[point], 0.1)

        # compute the required torque
        _, N = traj.shape
        required_torque = np.zeros((2,N))
        for i in range(N): 
            required_torque[:,i:i+1] = rbt.forward_dynamics(traj[:2,i:i+1], traj[2:4,i:i+1], traj[4:6,i:i+1])

        # create the dataset
        data = np.zeros((2*(N_samples+1), N))
        dist_bool = np.zeros((N_samples+1, N))
        i = 0
        
        for sample in range(N_samples+1):
            if sample == 0:
                data[i:i+2,:] = copy.deepcopy(required_torque) #insert the ground truth at the first row
                
            else:     
                data[i:i+2,:] = required_torque + np.random.normal(0, noise_level, (2,N))
                if sample == 2 :
                    ext_force = np.array([[2.],[0.]]).reshape(2,1)
                elif sample == 3:
                    ext_force = np.array([[1.],[1.]]).reshape(2,1)
                elif sample == 4:
                    ext_force = np.array([[0.],[-2.]]).reshape(2,1)
                else :
                    ext_force = np.array([[-1.],[1.]]).reshape(2,1)
                
                  
                disturbance_time = np.random.randint(low=0, high=N-20)
                force_ground_truth["traj_"+str(point)][sample-1,:] = np.array([ext_force[0,0], ext_force[1,0], disturbance_time])
                #print(f"I'm here point, {point}, sample {sample}, disturbance time {disturbance_time}")  
                for j in range(disturbance_time, disturbance_time+20):
                    
                    dist =rbt.forward_dynamics(traj[:2,j:j+1], traj[2:4,j:j+1], traj[4:6,j:j+1], ext_force)
                    #print(f"dist {dist.transpose()}")
                    data[i:i+2,j:j+1] = dist
                    if ext_force[0,0] != 0 or ext_force[1,0] != 0:
                        dist_bool[sample, j:j+1] = 1

                
                disturbance_ground_truth["traj_" + str(point)] = dist_bool
            dataset["traj_" + str(point)] = data
            i += 2
            
    return dataset, disturbance_ground_truth, force_ground_truth

if __name__ == 'main':
    
    rbt = Robot(1.,1.)
    print("HALO")
    ref = trajectories.createTrajectory(np.array([[0.1],[0.2]]), np.array([[0.2],[0.]]), 5., 0.1)
    plt.figure("Trajectory")
    plt.plot(ref[0,:], ref[1,:])
    plt.figure("Velocities")
    plt.plot(ref[2,:])
    plt.plot(ref[3,:])
    plt.grid()
    plt.show()
    #(rbt, np.array([[0.1, 0.2], [1., 0.]]), np.array([[0.2, 0. ], [0., 1.]]), np.array([5., 5.]), 5)
    