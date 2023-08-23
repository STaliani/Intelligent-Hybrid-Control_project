from robot import Robot
import numpy as np

rbt = Robot(1.,1.)

q   = np.array([0.1,0.2]).reshape(2,1)
qd  = np.array([-0.1,0.1]).reshape(2,1)
qdd = np.array([0.1,-0.1]).reshape(2,1)

tau = rbt.forward_dynamics(q,qd,qdd)
print(tau)