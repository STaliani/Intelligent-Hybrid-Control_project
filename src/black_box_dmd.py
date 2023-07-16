import numpy as np
import matplotlib.pyplot as plt
import scipy
from math import sqrt


class DMD:
    """Dynamic Mode Decomposition (DMD)
    This class implements the Dynamic Mode Decomposition algorithm to find a low-rank a low-rank approximation of an input matrix X.
    The method is based on eigenvalue decomposition, and it can be used to extract modes from both periodic or abrupt changes in system dynamics,
    such as those exhibited by dynamical systems with oscillating behavior like chaotic at tractors or turbulent flows.
    
    Attributes:
    svd_rank : int           rank truncation parameter for SVD compression of the input data
    exact : boolean          flag indicating whether to compute only approximate solution using truncated SVD
    opt : boolen             optimization flag that indicates if we want to use optimized version of DMD
    mode_index :             list indices of selected modes
    Atilde : array [n x n]   Low-dimensional representation of the input data obtained through DMD
    eigs : array[r], complex Eigenvalues associated with each eigenvector corresponding to the columns of `Atilde`
    Phi : array [m x r]      Matrix containing left singular vectors of `X`, where m = number of samples and r <= min(svd_rank, n), which are the effective dimensions after compressing them via SVD
    f : function handle      Function handle representing the time evolution operator acting on the original state space
    
    Methods:
    fit(): Fits the dynamic model onto the given dataset
    predict(): Predicts future states of the system under consideration
    reconstruct(): Reconstructs the full trajectory of the system starting from its initial condition
    plot_eigs(): Plots the top few dominant eigenmodes of the reduced Koopman operator
    _compute_phi(): Computes the left singular vectors of the compressed input data
    _build_lowdin_basis(): Builds the Lodin basis matrices required for computing the projection operators
    __init__(): Initializes all necessary attributes
    
    References:
    <NAME>, et al., "Dynamic mode decomposition: Data-driven modeling of complex
    systems" Physical Review E 96, no. 5 (2017):
    052408. https://journals.aps.org/pre/abstract
    """
    def __init__(self, svd_rank:int=None, exact:bool=False) -> None:
        self._svd_rank=svd_rank
        self._exact=exact
        self._opt=True #optimized implementation not available yet!
        self._mode_indices=[]
        self._Atilde=np.array([])
        self._eigs=np.array([], dtype='complex')
        self._Phi=np.array([[]])
        self._f=lambda t,x: np.zeros((len(t), len(x)))
        return
    
    def fit(self):
        Atilde=scipy.sparse.csc_matrix(_get_dynamics())#TODO implement this
        evals,evecs=scipy.sparse.linalg.eigsh(Atilde,-1)
        idx=evals.argsort()[::-1][:min(num_dominant_freq, Atilde.shape[0])]
        eigenvectors=[evecs[:,i].toarray().flatten()/sqrt(abs(eval)) for i in range(idx)]
        eigenvalues=[eval**2/(max(Atilde)-min(Atilde))]*len
        self._eigs=(sorted([(ev,ei) for ev,(ei,)in zip(evals[idx]**2/(max(Atilde)-min(Atilde)), eigenvectors)], key =(itemgetter(0))))
        self._Phi=np.hstack(([v/norm(v)**2 for v in eigenvectors]))
        
    def predict(self):
        pass

    def reconstruct(self):
        pass

    def plot_eigs(self):
        plt.plot(range(min(3*self._svd_rank+1)), abs([eig**2 if i<self._svd_rank else 0 for i, eig in enumerate sorted([(i, eigi) for i, eigi in zip(*sorted(((abs(ei)**2).real, ei) for ei in self._eigs))])]))
        plt.xlabel('Index $j$ of largest $\sigma$-eigenvalue', fontsize=1+'em')
        plt.ylabel('$|\lambda_{j}|^2$', fontsize=1+' em')
        plt.title("Top {} Dominant Eigenvalues".format ('Exact 'if self._exact==True else ''),fontsize = 1+' em')
        plt.show()
        return
    
    def _compute_phi():
        U,S,_ = scipy.linalg.svd(X,full_matrices=False)[:3]
        Phi=_compress_data(U[:,:r], X).T
        if r<m and m>p+q:
            warnings.warn("The SVD compression rank is less than optimal!")
        elif p==q or q>=n//2:#if there are more modes to be found
            print("Warning: The number of inputs exceeds twice that of outputs.")
        else:
            raise ValueError("SVD Rank should exceed both dimensions")
        return
    @staticmethod
    def _build_lowdin_basis(A):
        _, svals, Vh = scipy.sparse.linalg.svds(A, k, maxiter=maxitersize, which="LM", tol=tol)
        Smat = spdiags(svals**(-0.5), [0], A, B = dot(Vh.conj().T * sqrt(svals))
        basis = []
        for i in range(k):
            col = zeros(shape=(A.shape[0]))
            col[i::k]=sqrt(svals)[i]**0.5*ones(
            shape=[divmod(A.shape[0]+k-1, k)[0]])#normalize the columns so they have unit norm
            basis += [col]
        return array(basis).T


