import numpy as np

class DMD:
    def __init__(self):
        self.A_tilde = None
        self.Phi = None
        self.A = None

    def dmd(self, data: np.ndarray, r: int):
        """ Dynamic Mode Decomposition (DMD) algorithm.
            input:  data - data matrix
                    R - rank of the Koopman operator
            output: A_tilde - Koopman matrix
                    Phi - eigenvectors of the Koopman matrix
                    A - coefficient matrix
        """

        ## Build data matrices
        X1 = data[:, : -1]
        X2 = data[:, 1 :]
        ## Perform singular value decomposition on X1
        u, s, v = np.linalg.svd(X1, full_matrices = False)
        ## Compute the Koopman matrix
        self.A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
        ## Perform eigenvalue decomposition on A_tilde
        self.Phi, Q = np.linalg.eig(self.A_tilde)
        ## Compute the coefficient matrix
        Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
        self.A = Psi @ np.diag(self.Phi) @ np.linalg.pinv(Psi)

        return self.A_tilde, self.Phi, self.A

    def dmd4cast(self, data:np.array, r:int, pred_step:int):
        N, T = data.shape
        
        mat = np.append(data, np.zeros((N, pred_step)), axis = 1)
        for s in range(pred_step):
            mat[:, T + s] = (self.A @ mat[:, T + s - 1]).real
        return mat[:, - pred_step :]