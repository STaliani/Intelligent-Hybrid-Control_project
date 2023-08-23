import numpy as np 
import sympy as sp


def createTrajectory(start, goal, T, dT):
    N = round(T/dT) + 1
    # create traj 
    tau = sp.symbols('tau', real = True)
    Traj = np.array([tau, tau]).reshape(2,1)
    s = -2*tau**3 + 3*tau**2
    for i in range(2):
        traj = start[i,0] + s*(goal[i,0] - start[i,0])
        Traj[i] = traj
    
    dp = np.array([sp.diff(Traj[0,0])/T,sp.diff(Traj[1,0])/T]).reshape(2,1)
    ddp = np.array([sp.diff(dp[0,0])/T,sp.diff(dp[1,0])/T]).reshape(2,1)

    ref = np.zeros((6,N))
    j = 0
    for t in range(0,N,1):
        value = t/N
        ref[0:2, j] = np.array([Traj[0,0].subs(tau, value),Traj[1,0].subs(tau, value)]).transpose()
        ref[2:4, j] = np.array([dp[0,0].subs(tau, value),dp[1,0].subs(tau, value)]).transpose()
        ref[4:6, j] = np.array([ddp[0,0].subs(tau, value),ddp[1,0].subs(tau, value)]).transpose()
        j+=1

    return ref