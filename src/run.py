import numpy as np
from robot import Robot
import trajectories
import matplotlib.pyplot as plt

# initialize a 2R planar robot
rbt = Robot(1.,1.)

# initialize initial and final position for the robot
start_points = np.array([[0.1, 0.2], [1., 0.], [ 0.5, 0.5], [-1.1, 0.2], [0.8, -0.2]]).transpose()
end_points   = np.array([[0.2, 0. ], [0., 1.], [-0.5, 1.3], [-0.3,-1.1], [0.1, -0.7]]).transpose()

_, n = start_points.shape

for point in range(n):
    start_q = rbt.inverseKinimatics(start_points[0,point], start_points[1,point])
    end_q = rbt.inverseKinimatics(end_points[0,point], end_points[1,point])
    traj = trajectories.createTrajectory(start_q, end_q, 5, 0.1)
    _, N = traj.shape
    required_torque = np.zeros((2,N))
    for i in range(N): 
        required_torque[:,i:i+1] = rbt.forward_dynamics(traj[:2,i:i+1], traj[2:4,i:i+1], traj[4:6,i:i+1])

    plt.figure(str(point))
    plt.plot(np.rad2deg(traj[0,:]))
    plt.plot(np.rad2deg(traj[1,:]))
    plt.grid()
    title = "torque profile " + str(point)
    plt.figure(title)
    plt.plot(required_torque[0,:])
    plt.plot(required_torque[1,:])
    plt.grid()

#plt.show()