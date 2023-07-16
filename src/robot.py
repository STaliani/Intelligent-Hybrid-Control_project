import numpy as np

class Robot:
    def __init__(self, l1:float, l2:float):
        self.q1 
        self.q2
        self.x
        self.y
        self.l1 = l1
        self.l2 = l2
    
    def forwardKinematics(self)->None:
        self.x = self.l1*np.cos(self.q1) + self.l2*np.cos(self.q1 + self.q2)
        self.y = self.l1*np.sin(self.q1) + self.l2*np.sin(self.q1 + self.q2)
        
    def inverseKinimatics(self)->None:
        c2 = (self.x**2 + self.y**2 + self.l1**2 + self.l2**2)/(2*self.l1*self.l2)
        s2 = np.sqrt(1 - c2**2)
        self.q2 = np.arctan2(c2,s2)
        alpha = np.arctan2(self.x**2, self.y**2)
        c_beta = self.l1 + self.l2*c2
        s_beta = self.l2*s2
        beta = np.arctan2(c_beta,s_beta)
        self.q1 = alpha - beta

    def dynamics(self)->None:
        pass