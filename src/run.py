import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
from robot import Robot
import roboticstoolbox as rbtx
import trajectories as traj

## Create robot with its model ##
rbt = Robot(l1= 1, l2=1, m1=1, m2=1)

## crate trajectories ##
start_points = np.array([[0., 1.], [1., 0.], [ 0.5, 0.5], [-0.1, 0.2], [0.2, -0.2]]).transpose()
end_points   = np.array([[1., 0.], [0., 1.], [-0.5, 0.5], [-0.3,-0.1], [0.1, -0.1]]).transpose()

print(start_points.shape, end_points.shape)
