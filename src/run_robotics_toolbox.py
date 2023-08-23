import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
from robot import Robot
import roboticstoolbox as rbtx
import spatialmath as spm
import time
#import trajectories as traj
import swift
## Create robot with its model ##
#rbt = Robot(l1= 1, l2=1, m1=1, m2=1)
#env  = swift.Swift()
#panda = rbtx.models.DH.Panda()
#env.launch()
panda = rbtx.models.Panda()

## crate trajectories ##
# Create the start and end points
start_points = np.array([[0.1, 0.2, 0.2], [1., 0., 1.], [ 0.5, 0.5, 0.5], [-0.1, 0.2, 0.3], [0.2, -0.2, 0.2]])
start_rpy    = np.array([[0., 0., 0.], [0., np.pi/2, 0.], [ np.pi/4, 0., 0.], [np.pi/2, np.pi/4, 0.], [np.pi/4, np.pi/3, np.pi/6]])
end_points   = np.array([[0.2, 0., 1.], [0., 1., 0.], [-0.5, 0.5, 0.5], [-0.3,-0.1, 0.5], [0.1, -0.1, 0.2]])
end_rpy      = np.array([[0., 0., 0.], [0., np.pi/2, 0.], [ np.pi/4, 0., 0.], [np.pi/2, np.pi/4, 0.], [np.pi/4, np.pi/3, np.pi/6]])

m, _ = start_points.shape

for i in range(m): 
    start = spm.SE3(start_points[i,0],start_points[i,1],start_points[i,2])*spm.SE3.RPY(start_rpy[i,0],start_rpy[i,1],start_rpy[i,2])
    print('start:\n', start)
    end = spm.SE3(end_points[i,0],end_points[i,1],end_points[i,2])*spm.SE3.RPY(end_rpy[i,0],end_rpy[i,1],end_rpy[i,2])
    print('end:\n', end)
    
    sol = panda.ikine_LMS(start, tol= 1e-4)
    qi = sol.q
    print(qi)
    sol = panda.ikine_LMS(end  , q0=qi)
    qe = sol.q
    print(qe)
    traj = rbtx.jtraj(qi, qe, 50)
    traj1 = panda.jtraj(end, 50, 5)
    print('traj size', traj.q.shape)
    print('traj type', type(traj.q))
    print('Traj poiint', traj.q[0,:])
    T = [panda.fkine(qi) for qi in traj.q]
    tau = panda.rne(traj.q, traj.qd, traj.qdd)
    print(tau)
    #tau = np.zeros((50, 7))
    #for i in range(50):
    #    tau[i,:]  = panda.rne(q=traj.qd[i,:], qd= traj.qd[i,:], qdd=traj.qdd[i,:])
    #    panda.rne()
    #print(tau)
    #env.add(panda)
    #time.sleep(3)
    #for qk in traj.q:             # for each joint configuration on trajectory
    #    panda.q = qk 
    #    time.sleep(0.1)
    #    env.step() 
    #env.remove(panda)
    #panda.plot(traj.q, backend='pyplot')


    