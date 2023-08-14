import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
from robot import Robot
import roboticstoolbox as rbtx
import spatialmath as spm
#import trajectories as traj

## Create robot with its model ##
#rbt = Robot(l1= 1, l2=1, m1=1, m2=1)
panda = rbtx.models.DH.Panda()

#print(panda.dynamics)


## crate trajectories ##
# Create the start and end points
start_points = np.array([[0., 1., 0.], [1., 0., 1.], [ 0.5, 0.5, 0.5], [-0.1, 0.2, 0.3], [0.2, -0.2, 0.2]])
start_rpy    = np.array([[0., 0., 0.], [0., np.pi/2, 0.], [ np.pi/4, 0., 0.], [np.pi/2, np.pi/4, 0.], [np.pi/4, np.pi/3, np.pi/6]])
end_points   = np.array([[1., 0., 1.], [0., 1., 0.], [-0.5, 0.5, 0.5], [-0.3,-0.1, 0.5], [0.1, -0.1, 0.2]])
end_rpy      = np.array([[0., 0., 0.], [0., np.pi/2, 0.], [ np.pi/4, 0., 0.], [np.pi/2, np.pi/4, 0.], [np.pi/4, np.pi/3, np.pi/6]])

m, _ = start_points.shape

for i in range(m): 
    rpy   = spm.SO3.RPY(start_rpy[i,0], start_rpy[i,1], start_rpy[i,2])
    start = spm.SE3(rpy.A, start_points[i,:])
    rpy   = spm.SO3.RPY(end_rpy[i,0], end_rpy[i,1], end_rpy[i,2])
    end   = spm.SE3(rpy.A, end_points[i,:])
    
    qi = panda.ik_nr(start, np.zeros(7))
    qe = panda.ik_nr(end  , qi)

    traj = rbtx.jtraj(qi, qe, 50)
    panda.plot(traj)
    print('halo')


    