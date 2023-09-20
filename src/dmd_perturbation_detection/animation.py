from .robot import Robot
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
def update(i):
    ax.clear()
    x1 = round(rbt.l1*np.cos(q[0,i]),2)
    y1 = round(rbt.l1*np.sin(q[0,i]),2)
    x2 = x1 + round(rbt.l2*np.cos(q[0,i]+q[1,i]),2)
    y2 = y1 + round(rbt.l2*np.sin(q[0,i]+q[1,i]),2)

    lx1 = [0]
    ly1 = [0]
    lx2 = [x1]
    ly2 = [y1]
    for j in range(11):
        lx1.append(x1/10*j)
        ly1.append(y1/10*j)
        lx2.append(x1 + (x2-x1)/10*j)
        ly2.append(y1 + (y2-y1)/10*j)
    ax.plot(lx1, ly1, 'b')
    ax.plot(lx2, ly2, 'r')
    
    if bool(disturbance[i]):
        ax.plot(x2, y2, 'ro')
    else:
        ax.plot(x2, y2, 'go')
        
    if i> f[2] and i< f[2]+20:
        ax.quiver(x2, y2, f[0], f[1], color='k', scale=5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid()
    ax.set_xlabel("x [dm]")
    ax.set_ylabel("y [dm]")
    plt.draw()
    
    
def animate(my_rbt:Robot, q_ref:np.ndarray, measured_disturbance:np.ndarray, force: np.array, dt:float, N_samples:int)->FuncAnimation:
    global ax, rbt, q, disturbance, f
    f = force
    disturbance = measured_disturbance
    rbt = my_rbt
    q = q_ref
    fig = plt.figure("Robot animation")
    ax = fig.add_subplot()

    anim = FuncAnimation(fig, update, interval= 50,frames=range(q_ref.shape[1]), repeat=False)
    plt.show()
    return anim