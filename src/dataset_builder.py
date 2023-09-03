import trajectories
from robot import Robot
import numpy as np
import matplotlib.pyplot as plt

def create_dataset(rbt: Robot, start_points:np.ndarray, end_points:np.ndarray, execution_time:np.ndarray, N_samples:int, noise_level:float = 0.2)->dict:
    _, n = start_points.shape
    dataset = dict.fromkeys(["traj_" + str(key) for key in range(N_samples)])
    for point in range(n):
        start_q = rbt.inverseKinimatics(start_points[0,point], start_points[1,point])
        end_q = rbt.inverseKinimatics(end_points[0,point], end_points[1,point])
        traj = trajectories.createTrajectory(start_q, end_q, execution_time[point], 0.1)
        plt.figure("Velocities")
        plt.plot(traj[2,:])
        plt.plot(traj[3,:])
        plt.grid()
        _, N = traj.shape
        required_torque = np.zeros((2,N))
        for i in range(N): 
                    required_torque[:,i:i+1] = rbt.forward_dynamics(traj[:2,i:i+1], traj[2:4,i:i+1], traj[4:6,i:i+1])
        plt.figure("Torques")
        plt.plot(required_torque[0,:])
        plt.plot(required_torque[1,:])
        plt.grid()
        #plt.show()
        data = np.zeros((2*(N_samples+1), N))
        i = 0
        for sample in range(N_samples+1):
            if sample == 0:
                data[i:i+2,:] = required_torque #insert the ground truth at the first row
            else:     
                if sample == 2 :
                    ext_force = np.array([[1.],[0.]]).reshape(2,1)
                elif sample == 4:
                    ext_force = np.array([[0.],[1.]]).reshape(2,1)
                else :
                    ext_force = np.array([[0.],[0.]]).reshape(2,1)
                disturbance_time = np.random.randint(0, N-10)
                for j in range(disturbance_time, disturbance_time+10):
                    required_torque[:,j:j+1] = rbt.forward_dynamics(traj[:2,j:j+1], traj[2:4,j:j+1], traj[4:6,j:j+1], ext_force)
                data[i:i+2,:] = required_torque + np.random.normal(0, noise_level, (2,N))
            dataset["traj_" + str(point)] = data
            i += 2
            
    return dataset

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
    