import numpy as np 
import simpy as sp


def createTrajectory(start, goal, T, dT, plus, rbt):
    N = round(T/dT) + 1
    # create traj 
    tau = sp.symbols('tau', real = True)
    Traj = np.array([tau, tau, tau]).reshape(3,1)
    s = -2*tau**3 + 3*tau**2
    for i in range(3):
        traj = start[i,0] + s*(goal[i,0] - start[i,0])
        Traj[i] = traj
    
    dp = np.array([sp.diff(Traj[0,0])/T,sp.diff(Traj[1,0])/T,sp.diff(Traj[2,0])/T]).reshape(3,1)
    ddp = np.array([sp.diff(dp[0,0])/T,sp.diff(dp[1,0])/T,sp.diff(dp[2,0])/T]).reshape(3,1)

    ref = np.zeros((9,N))
    force_ref = np.zeros((3,N))
    j = 0
    for t in range(0,N,1):
        value = t/N
        des_acc = np.array([ddp[0,0].subs(tau, value),ddp[1,0].subs(tau, value),ddp[2,0].subs(tau, value)]).transpose()
        ref[0:3, j] = np.array([Traj[0,0].subs(tau, value),Traj[1,0].subs(tau, value),Traj[2,0].subs(tau, value)]).transpose()
        ref[3:6, j] = np.array([dp[0,0].subs(tau, value),dp[1,0].subs(tau, value),dp[2,0].subs(tau, value)]).transpose()
        ref[6:10, j] = des_acc
        force_ref[:, j] = rbt.mass*des_acc
        j+=1

    return ref, force_ref