import numpy as np

class DMD:
    def __init__(self):
        pass
    
    def dmd(self, X: np.ndarray, X_prime: np.ndarray, r:int):
        U, Sigma, V = np.linalg.svd(X) #step 1
        Ur = U[:,:r]
        Sigmar = Sigma[:r, :r]
        Vr = V[:,:r]

        Atilde = Ur.transpose()@X_prime@V/Sigmar
        W, Lambda = np.linalg.eig(Atilde)
        Phi = X_prime@(Vr/Sigmar)@W
        alpha1 = Sigmar@Vr[0,:].transpose()
        b = np.linalg.lstsq(W@Lambda,alpha1,rcond=None)[0]
        return Phi, Lambda, b
    

