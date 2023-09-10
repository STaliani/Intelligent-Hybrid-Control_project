import numpy as np
import matplotlib.pyplot as plt
class Robot:

    g = 9.81
    def __init__(self, l1:float, l2:float, q1:float=0, q2:float=0, m1:float=0.5, m2:float=0.5, dt:float=0.1):
        self.q1 = q1 
        self.q2 = q2
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.I1 = 0.5*m1*l1**2
        self.I2 = 0.5*m2*l2**2
        self.dt = dt
        self.forwardKinematics()
        print("Robot created")
        
    def forwardKinematics(self, q:np.ndarray=None)->np.ndarray:
        if q is None:
            self.x = self.l1*np.cos(self.q1) + self.l2*np.cos(self.q1 + self.q2)
            self.y = self.l1*np.sin(self.q1) + self.l2*np.sin(self.q1 + self.q2)
        else:
            pos = np.zeros((2,q.shape[1]))
            for i in range(q.shape[1]):
                pos[0,i] = self.l1*np.cos(q[0,i]) + self.l2*np.cos(q[0,i] + q[1,i])
                pos[1,i] = self.l1*np.sin(q[0,i]) + self.l2*np.sin(q[0,i] + q[1,i])    
            return pos
        
    def inverseKinimatics(self, x:float = None, y:float= None)->np.ndarray:
        if x == None and y==None:
            x = self.x
            y = self.y
        c2 = (x**2 + y**2 - self.l1**2 - self.l2**2)/(2*self.l1*self.l2)
        if c2 > 1:
            raise Exception("The point is out of the workspace")
        s2 = np.sqrt(1 - c2**2)
        self.q2 = np.arctan2(s2,c2)
        alpha = np.arctan2(y, x)
        c_beta = self.l1 + self.l2*c2
        s_beta = self.l2*s2
        beta = np.arctan2(s_beta,c_beta)
        self.q1 = alpha - beta
        return np.array([self.q1, self.q2]).reshape(2,1)
        
    def forward_dynamics(self, q, dq, ddq, external_force = np.zeros((2,1)))->np.ndarray:
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        q1 = q[0,0]
        q2 = q[1,0]
        I1 = self.I1
        I2 = self.I2
        
        M = np.array([[m1*l1**2 + I1 + m2*((l1/2)**2 + l2**2 + 2*(l1/2)*l2*np.cos(q2)) + l2, m2*(l2**2 + (l1/2)*l2*np.cos(q2))+l2],
                      [m2*(l2**2 + (l1/2)*l2*np.cos(q2)+l2)                                , m2*l2**2 + I2                       ]])
        C = np.array([[-m2*(l1/2)*l2*np.sin(q2)*dq[0,0], -m2*(l1/2)*l2*np.sin(q2)*(dq[0,0]+ dq[1,0])],
                      [ m2*(l1/2)*l2*np.sin(q2)*dq[1,0],  0                                     ]])
        G = np.array([[(m1*l1 + m2*(l1/2))*self.g*np.cos(q1) + m2*self.g*l2*np.cos(q1+q2)],
                      [m2*self.g*l2*np.cos(q1+q2)                                        ]])
        
        external_force_compensation = self.jacobian(q).T@external_force
        
        tau = M@ddq + C@dq + G - external_force_compensation
        
        return tau
    
    def inverse_dynamics(self, q_init: np.ndarray, dq_init: np.ndarray, tau:np.ndarray, external_force = np.zeros((2,1)))->np.ndarray:
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        I1 = self.I1
        I2 = self.I2
        
        _, N = tau.shape
        q = np.zeros((2,N+1))
        dq = np.zeros((2,N+1))
        ddq = np.zeros((2,N+1))
        q[:,0] = q_init[:,0]
        dq[:,0] = dq_init[:,0]
        for i in range(N):
            q1 = q[0,i]
            q2 = q[1,i]
            dq1 = dq[0,i]
            dq2 = dq[1,i]
            
            print("dq1: ", dq1)
            print("dq2: ", dq2)
            
            M = np.array([[m1*l1**2 + I1 + m2*((l1/2)**2 + l2**2 + 2*(l1/2)*l2*np.cos(q2)) + l2, m2*(l2**2 + (l1/2)*l2*np.cos(q2))+l2],
                          [m2*(l2**2 + (l1/2)*l2*np.cos(q2)+l2)                                , m2*l2**2 + I2                       ]])
            C = np.array([[-m2*(l1/2)*l2*np.sin(q2)*dq1, -m2*(l1/2)*l2*np.sin(q2)*(dq1+ dq2)],
                          [ m2*(l1/2)*l2*np.sin(q2)*dq2,  0                                     ]])
            G = np.array([[(m1*l1 + m2*(l1/2))*self.g*np.cos(q1) + m2*self.g*l2*np.cos(q1+q2)],
                          [m2*self.g*l2*np.cos(q1+q2)                                        ]])
            
            print("M: \n", M)
            print("C: \n", C)
            print("G: \n", G)
            ddq[:,i:i+1] = np.linalg.inv(M)@(tau[:,i:i+1] - C@dq[:,i:i+1] - G + external_force)
            dq[:,i+1] = dq[:,i] + ddq[:,i]*self.dt
            q[:,i+1] = q[:,i] + dq[:,i]*self.dt + 0.5*ddq[:,i]*self.dt**2
        
        return q, dq, ddq
    
    def jacobian(self,q):
        if q.shape != (2,1):
            q.reshape(2,1)
        J = np.array([[-self.l1*np.sin(q[0,0]) - self.l2*np.sin(q[0,0]+q[1,0]), -self.l2*np.sin(q[0,0]+q[1,0])] , 
                      [ self.l1*np.cos(q[0,0]) + self.l2*np.cos(q[0,0]+q[1,0]),  self.l2*np.cos(q[0,0]+q[1,0])]])
        
        return J
    
    def plot_robot(self, q:np.ndarray=None):
        if q is None:
            self.forwardKinematics()
            q = np.array([self.q1, self.q2]).reshape(2,1)
        
        x1 = round(self.l1*np.cos(q[0,0]),2)
        y1 = round(self.l1*np.sin(q[0,0]),2)
        x2 = x1 + round(self.l2*np.cos(q[0,0]+q[1,0]),2)
        y2 = y1 + round(self.l2*np.sin(q[0,0]+q[1,0]),2)
        print("x1, x2: ", x1, x2)
        print("y1, y2: ", y1, y2)
        lx1 = [0]
        ly1 = [0]
        lx2 = [x1]
        ly2 = [y1]
        for i in range(11):
            lx1.append(x1/10*i)
            ly1.append(y1/10*i)
            lx2.append(x1 + (x2-x1)/10*i)
            ly2.append(y1 + (y2-y1)/10*i)
        plt.figure("Robot")
        plt.plot(lx1, ly1, 'r')
        plt.plot(lx2, ly2, 'b')
        #plt.quiver(0, 0, x1, y1, colour='r', linewidths=2)
        #plt.quiver(x1, y1, x2, y2, colour='b', linewidths=2)
        plt.plot(x2, y2, 'o')
        plt.xlabel("x [dm]")
        plt.ylabel("y [dm]")
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.grid()
        plt.show()
    
if __name__ == '__main__':
    
    rbt = Robot(1.,1.)
    rbt.plot_robot()
    